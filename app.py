"""
Streamlit Web Application: Agentic RAG-Based AI Research Assistant
Capstone Project - Natural Language Processing
Powered by Local Ollama (llama3) & Native LangChain Tool Calling
Features: Native Tool Calling, Topic Modeling (BERTopic), RAG Retriever (k=4),
Sentiment Analysis (TextBlob), Named Entity Recognition (spaCy), and Multilingual Translation.
"""

import os
import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from textblob import TextBlob
import spacy
from deep_translator import GoogleTranslator
from bertopic import BERTopic

# LangChain, Chroma & Local Ollama Native Tool Calling
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Graceful import for Embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent


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
    "**A College Capstone NLP Project** featuring **Native Tool Calling**, "
    "**RAG (k=4)**, **BERTopic Topic Modeling**, **ChromaDB**, and **Local LLM (Ollama - LLaMA 3)**."
)

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
    """Extracts and sanitizes text from an uploaded PDF stream."""
    reader = PdfReader(uploaded_file)
    pages_text = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages_text.append(re.sub(r'\s+', ' ', txt).strip())
    return "\n\n".join(pages_text)


def create_vectorstore_from_text(text: str) -> Chroma:
    """Chunks text and generates dense embeddings indexed inside ChromaDB."""
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


# 4. Agent Tool Definitions & Native Agent Creation
def build_agent_tools(vectorstore):
    """
    Constructs the tool catalog for the Native Tool Calling Agent:
    1. Document_Search: RAG retriever with top-4 chunks (k=4) & fallback.
    2. Topic_Modeling: BERTopic theme discovery.
    3. Sentiment_Analysis: TextBlob polarity & subjectivity.
    4. Info_Extraction: spaCy token-level Named Entity Recognition.
    5. Translate: deep-translator multilingual engine.
    """
    # Requirement 3: Fetch top 4 chunks (k=4)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def run_doc_search(query: str) -> str:
        clean_q = query.strip().strip('"').strip("'")
        docs = retriever.invoke(clean_q)
        if not docs:
            # Requirement 3: Explicit fallback string
            return "No directly matching excerpt found in the document."
        return "\n\n".join([f"[Excerpt {i+1}]: {d.page_content}" for i, d in enumerate(docs)])

    def run_topic_modeling(text: str) -> str:
        # Requirement 2: Topic Modeling using BERTopic on sentences
        try:
            # If input is very short, pull context from vectorstore if available
            clean_text = text.strip()
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', clean_text) if len(s.strip()) > 15]
            
            # If the user passed a query rather than text, retrieve top document passages
            if len(sentences) < 3:
                sample_docs = retriever.invoke("main research topics, methodology, and conclusion")
                if sample_docs:
                    clean_text = " ".join([d.page_content for d in sample_docs])
                    sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', clean_text) if len(s.strip()) > 15]

            if len(sentences) < 3:
                return "The provided text is too brief for statistical topic clustering (requires at least 3 sentences)."

            topic_model = BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                min_topic_size=2,
                verbose=False
            )
            topics, _ = topic_model.fit_transform(sentences)
            topic_info = topic_model.get_topic_info()

            # Return top 5 topic names
            top_topics = []
            for _, row in topic_info.head(5).iterrows():
                name = row.get("Name", f"Topic {row['Topic']}")
                top_topics.append(f"• {name}")

            return "Top Discovered Themes / Topics:\n" + "\n".join(top_topics) if top_topics else "No distinct topics found."
        except Exception as e:
            return f"Topic modeling error: {str(e)}"

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
            description="Use this tool to find specific facts, methodologies, definitions, or experimental data. Do NOT use this to count topics or summarize themes."
        ),
        Tool(
            name="Topic_Modeling",
            func=run_topic_modeling,
            description="Use this tool ONLY when the user asks about the main themes, topics, or clusters inside the document. Input must be a large chunk of text."
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


def create_agent(vectorstore) -> AgentExecutor:
    """
    Creates and returns a Native Tool Calling AgentExecutor powered by local Ollama (llama3).
    Includes safety limits: max_iterations=4, max_execution_time=60, early_stopping_method='generate'.
    """
    # Requirement 1 & 5: Local Ollama LLM with temperature=0
    llm = ChatOllama(
        model="llama3",
        temperature=0.0
    )

    # Build Tools (including Topic_Modeling & Document_Search with k=4)
    tools = build_agent_tools(vectorstore)

    # Requirement 1: Native Tool Calling ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert AI Research Assistant specializing in Natural Language Processing.\n"
            "You have access to specialized tools to answer inquiries accurately:\n"
            "- Use 'Document_Search' to retrieve specific factual information, methodology, or metrics.\n"
            "- Use 'Topic_Modeling' when the user asks about main themes, topics, or clusters in the paper.\n"
            "- Use 'Sentiment_Analysis' to evaluate emotional polarity and subjectivity.\n"
            "- Use 'Info_Extraction' to extract named entities (authors, universities, dates).\n"
            "- Use 'Translate' to translate non-English inputs into English.\n"
            "Always invoke the appropriate tool before responding."
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # Requirement 1: Native Tool Calling Agent
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    # Requirement 4: AgentExecutor with safety limits
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=4,
        max_execution_time=60,
        early_stopping_method="generate",
        handle_parsing_errors="Format error! You must use the provided tools correctly.",
        return_intermediate_steps=True
    )

# Alias for backward compatibility
initialize_ollama_agent = create_agent


# 5. Session State Management
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to your Agentic RAG AI Research Assistant!**\n\n"
                "• **Local LLM:** Ollama (`llama3`) via **Native Tool Calling** (JSON Schema)\n"
                "• **Capabilities:** RAG Document Retrieval (k=4), BERTopic Topic Modeling, Sentiment Analysis, Named Entity Recognition, Translation.\n\n"
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


# 6. Sidebar Controls & Demonstration Options
with st.sidebar:
    st.header("⚙️ Project Controls")
    st.markdown('<span class="metric-badge">LOCAL LLM: llama3</span><span class="metric-badge">TOOL CALLING</span>', unsafe_allow_html=True)
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
                        st.session_state.agent = create_agent(vectorstore)
                        st.session_state.processed_file = uploaded_pdf.name
                        
                        st.success(f"Successfully indexed: `{uploaded_pdf.name}`")
                        st.info("Native Tool Calling Agent initialized with `llama3` & ChromaDB (k=4)!")
                except Exception as e:
                    st.error(f"Error during document processing: {str(e)}")
        else:
            st.success(f"Active Paper: `{uploaded_pdf.name}`")
            st.caption("Agent is online and ready for queries.")

    st.markdown("---")
    st.subheader("🔍 2. Evaluator Demonstration")
    
    show_thoughts = st.toggle(
        "Show Agent Thoughts",
        value=True,
        help="Display the step-by-step native tool invocations and intermediate observations."
    )

    st.markdown("---")
    st.subheader("👥 Group Division")
    st.markdown("""
    - **Student 1:** NLP Backend: Jupyter Notebook, chunking, embeddings, Vector DB setup, classic NLP tools.
    - **Student 2:** Frontend Streamlit app, Native Tool Calling Agent, Ollama LLM integration, BERTopic Tool, testing.
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


# 7. Main Chat Area
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display Agent Thoughts if available and toggle is active
        if show_thoughts and msg.get("thoughts"):
            with st.expander("🧠 Agent Tool Calling & Execution Trace", expanded=False):
                for step_num, (action, observation) in enumerate(msg["thoughts"], 1):
                    st.markdown(f"**Step {step_num}: Tool Invoked: `{action.tool}`**")
                    st.markdown(f"*Input to Tool:* `{action.tool_input}`")
                    st.markdown(f"**Observation from Tool:**")
                    st.info(observation)

# User Chat Input
user_query = st.chat_input("Ask about facts, methodologies, main themes/topics, sentiment, or entities...")

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
            with st.spinner("🤖 Ollama Agent (llama3) is executing native tool calling..."):
                try:
                    response = st.session_state.agent.invoke({"input": user_query})
                    output_text = response.get("output", "")
                    intermediate_steps = response.get("intermediate_steps", [])
                    
                    # Graceful synthesis if early stopping triggered
                    if not output_text or "Agent stopped due to iteration limit" in output_text:
                        if intermediate_steps:
                            last_obs = intermediate_steps[-1][1]
                            output_text = f"Based on the analysis:\n\n{last_obs}"
                        elif st.session_state.vectorstore:
                            fallback_docs = st.session_state.vectorstore.similarity_search(user_query, k=2)
                            output_text = "\n\n".join([d.page_content for d in fallback_docs])
                        else:
                            output_text = "The assistant could not complete the request. Please refine your query."

                    # Display intermediate steps if toggle is active
                    if show_thoughts and intermediate_steps:
                        with st.expander("🧠 Agent Tool Calling & Execution Trace", expanded=True):
                            for step_num, (action, observation) in enumerate(intermediate_steps, 1):
                                st.markdown(f"**Step {step_num}: Selected Tool: `{action.tool}`**")
                                st.markdown(f"*Tool Input:* `{action.tool_input}`")
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
