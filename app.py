"""
Streamlit Web Application: Agentic RAG-Based AI Research Assistant
Capstone Project - Natural Language Processing
Powered by Local Ollama (llama3) & LangChain
Features: RAG, Agents, Tools, Vector DB (Chroma), Local LLM (Ollama), and Agent Thought Tracking
"""

import os
import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from textblob import TextBlob
import spacy
from deep_translator import GoogleTranslator

# LangChain, Chroma & Local Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Graceful import for Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent


# 1. Page Configuration
st.set_page_config(
    page_title="Agentic RAG AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application Header
st.title("🔬 Agentic RAG-Based AI Research Assistant")
st.markdown(
    "**A College Capstone NLP Project** combining **RAG**, **ReAct Agents**, "
    "**Custom NLP Tools**, **ChromaDB (Vector DB)**, and **Local LLM (Ollama - LLaMA 3)**."
)

# Custom Styling for Academic Presentation
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        background-color: #E0E7FF;
        color: #3730A3;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


# 2. Lazy Loading & Caching of spaCy Model
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()


# 3. Document Extraction & Preprocessing (CO1 - Practical 1)
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages_text.append(re.sub(r'\s+', ' ', txt).strip())
    return "\n\n".join(pages_text)


def create_vectorstore_from_text(text: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    docs = splitter.create_documents([text])
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return Chroma.from_documents(documents=docs, embedding=embeddings)


# 4. Agent Tool Catalog (Practical 2, 3, 5)
def build_agent_tools(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    def run_doc_search(query: str) -> str:
        # Sanitize query from extra quotes or whitespace that LLaMA 3 might pass
        clean_q = query.strip().strip('"').strip("'")
        docs = retriever.invoke(clean_q)
        if not docs:
            return "No directly matching excerpt found in the uploaded research paper."
        return "\n\n".join([f"[Excerpt {i+1}]: {d.page_content}" for i, d in enumerate(docs)])

    def run_sentiment(text: str) -> str:
        blob = TextBlob(text)
        pol = round(blob.sentiment.polarity, 4)
        subj = round(blob.sentiment.subjectivity, 4)
        label = "Positive" if pol > 0.1 else ("Negative" if pol < -0.1 else "Neutral")
        return f"Sentiment: {label} (Polarity Score: {pol}, Subjectivity Score: {subj})"

    def run_info_extraction(text: str) -> str:
        doc = nlp(text)
        ents = [f"{e.text} [{e.label_}]" for e in doc.ents]
        return "; ".join(ents) if ents else "No named entities recognized in text."

    def run_translation(query: str) -> str:
        clean_q = query.strip().strip('"').strip("'")
        translator = GoogleTranslator(source="auto", target="en")
        return translator.translate(clean_q)

    return [
        Tool(
            name="Document_Search",
            func=run_doc_search,
            description="Use this tool to search facts, methodologies, definitions, and experimental data inside the uploaded research paper."
        ),
        Tool(
            name="Sentiment_Analysis",
            func=run_sentiment,
            description="Use this tool to analyze the emotional polarity (-1 to +1) and subjective tone of any scientific passage or claim."
        ),
        Tool(
            name="Info_Extraction",
            func=run_info_extraction,
            description="Use this tool to extract key Named Entities (Persons, Organizations, Universities, Dates) from text."
        ),
        Tool(
            name="Translate",
            func=run_translation,
            description="Use this tool to translate non-English queries or excerpts into English."
        )
    ]


# 5. Local Ollama ReAct Agent Construction (CO4 - Practical 6 & 7)
def initialize_ollama_agent(vectorstore) -> AgentExecutor:
    # 1. Local LLM via Ollama
    llm = ChatOllama(
        model="llama3",
        temperature=0.0
    )

    # 2. Build Tools
    tools = build_agent_tools(vectorstore)

    # 3. Prompt Engineering: Calibrated ReAct Template for Local LLaMA 3
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

CRITICAL RULES:
1. When you receive an Observation that answers the Question, do NOT invoke any more tools. You MUST immediately write:
Thought: I now know the final answer
Final Answer: [your complete answer based on the observation]
2. Do NOT repeat the same tool call with the same input.
3. If the user asks a greeting or general question, write 'Thought: I now know the final answer' and give the Final Answer directly.
4. Never invent tool names outside of [{tool_names}].

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # 4. Construct Agent & AgentExecutor with high iterations & early_stopping_method='generate'
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors="Check your output format! If you have the answer, write 'Thought: I now know the final answer' followed by 'Final Answer: <your answer>'.",
        return_intermediate_steps=True,
        max_iterations=10,
        max_execution_time=120,
        early_stopping_method="generate"
    )


# 6. Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to your Agentic RAG AI Research Assistant!**\n\n"
                "• **Local LLM Engine:** Ollama (`llama3`)\n"
                "• **Key Topics:** RAG, ReAct Agent, 4 NLP Tools, ChromaDB Vector Store.\n\n"
                "Please upload a research paper (PDF) from the left sidebar to activate the agent."
            ),
            "thoughts": None
        }
    ]

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# 7. Sidebar Controls & Demonstration Options
with st.sidebar:
    st.header("⚙️ Project Controls")
    st.markdown('<span class="metric-badge">LOCAL LLM: llama3</span><span class="metric-badge">RAG ACTIVE</span>', unsafe_allow_html=True)
    st.caption("College Capstone NLP Project • Group of 2 Students")

    st.markdown("---")
    st.subheader("📄 1. Document Ingestion (RAG)")
    uploaded_pdf = st.file_uploader(
        "Upload Academic Paper (PDF)",
        type=["pdf"],
        help="Upload an academic research PDF for automated chunking and Chroma vectorization."
    )

    if uploaded_pdf is not None:
        if st.session_state.processed_file != uploaded_pdf.name:
            with st.spinner("Processing PDF: Extracting text, chunking & creating ChromaDB vector store..."):
                try:
                    raw_text = extract_text_from_pdf(uploaded_pdf)
                    
                    if not raw_text.strip():
                        st.error("No extractable text found in the uploaded PDF.")
                    else:
                        vectorstore = create_vectorstore_from_text(raw_text)
                        st.session_state.vectorstore = vectorstore
                        st.session_state.agent = initialize_ollama_agent(vectorstore)
                        st.session_state.processed_file = uploaded_pdf.name
                        
                        st.success(f"Successfully indexed: `{uploaded_pdf.name}`")
                        st.info("ReAct Agent initialized with `llama3` & ChromaDB retriever!")
                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")
        else:
            st.success(f"Active Paper: `{uploaded_pdf.name}`")
            st.caption("Agent is online and ready for queries.")

    st.markdown("---")
    st.subheader("🔍 2. Evaluator Demonstration")
    
    # Toggle for showing Agent Thoughts
    show_thoughts = st.toggle(
        "Show Agent Thoughts",
        value=True,
        help="Display the step-by-step ReAct reasoning trace, tool invocations, and observations."
    )

    st.markdown("---")
    st.subheader("👥 Group Division")
    st.markdown("""
    - **Student 1:** NLP Backend: Jupyter Notebook, chunking, embeddings, Vector DB setup, classic NLP tools.
    - **Student 2:** Frontend Streamlit app, Agent setup, Ollama LLM integration, Prompt Engineering, testing.
    """)

    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Conversation cleared. Ready for your next research inquiry!",
                "thoughts": None
            }
        ]
        st.rerun()


# 8. Main Chat Area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Agent Thoughts if available and toggle is active
        if show_thoughts and msg.get("thoughts"):
            with st.expander("🧠 Agent Reasoning & Tool Execution Steps", expanded=False):
                for step_num, (action, observation) in enumerate(msg["thoughts"], 1):
                    st.markdown(f"**Step {step_num}: Tool Invoked: `{action.tool}`**")
                    st.markdown(f"*Input to Tool:* `{action.tool_input}`")
                    st.code(action.log, language="text")
                    st.markdown(f"**Observation from Tool:**")
                    st.info(observation)

# User Chat Input
user_query = st.chat_input("Ask a question about the document, analyze sentiment, or extract entities...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query, "thoughts": None})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.agent is None:
            warning_text = "⚠️ Please upload an academic research paper (PDF) from the sidebar first to activate the Agent."
            st.warning(warning_text)
            st.session_state.messages.append({"role": "assistant", "content": warning_text, "thoughts": None})
        else:
            with st.spinner("🤖 Ollama Agent (llama3) is reasoning and selecting tools..."):
                try:
                    response = st.session_state.agent.invoke({"input": user_query})
                    output_text = response.get("output", "")
                    intermediate_steps = response.get("intermediate_steps", [])
                    
                    # Graceful fallback if agent stopped due to iteration limit without output
                    if not output_text or "Agent stopped due to iteration limit" in output_text:
                        if intermediate_steps:
                            # Synthesize an answer from the observations collected
                            last_obs = intermediate_steps[-1][1]
                            output_text = f"Based on the document excerpts retrieved:\n\n{last_obs}"
                        elif st.session_state.vectorstore:
                            fallback_docs = st.session_state.vectorstore.similarity_search(user_query, k=2)
                            output_text = "\n\n".join([d.page_content for d in fallback_docs])
                        else:
                            output_text = "The research assistant was unable to finalize the answer within the time limit. Please rephrase your query."

                    # Display Agent Thoughts live if toggle is active
                    if show_thoughts and intermediate_steps:
                        with st.expander("🧠 Agent Reasoning & Tool Execution Steps", expanded=True):
                            for step_num, (action, observation) in enumerate(intermediate_steps, 1):
                                st.markdown(f"**Step {step_num}: Selected Tool: `{action.tool}`**")
                                st.markdown(f"*Tool Input:* `{action.tool_input}`")
                                st.code(action.log, language="text")
                                st.markdown(f"**Tool Observation:**")
                                st.info(observation)

                    # Display final answer
                    st.markdown(output_text)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": output_text,
                        "thoughts": intermediate_steps if intermediate_steps else None
                    })
                except Exception as err:
                    # In case of any unhandled execution exception, retrieve context safely
                    error_str = str(err)
                    if "iteration limit" in error_str.lower() and st.session_state.vectorstore:
                        fallback_docs = st.session_state.vectorstore.similarity_search(user_query, k=2)
                        fallback_answer = "Here is the relevant information retrieved from the paper:\n\n" + "\n\n".join([f"• {d.page_content}" for d in fallback_docs])
                        st.markdown(fallback_answer)
                        st.session_state.messages.append({"role": "assistant", "content": fallback_answer, "thoughts": None})
                    else:
                        error_msg = f"❌ Execution Error: {error_str}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "thoughts": None})
