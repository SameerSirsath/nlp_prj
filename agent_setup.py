"""
Agentic Workflow & Native Tool Calling Agent Setup
Capstone Project - Natural Language Processing
Powered by Local Ollama (llama3) & Native Tool Calling
"""

import os
import re
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from typing import Optional, List, Dict, Any

from textblob import TextBlob
import spacy
from deep_translator import GoogleTranslator
from bertopic import BERTopic

# Local Ollama & LangChain Native Tool Calling
from langchain_ollama import ChatOllama
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# Lazy load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def build_agent_tools(vectorstore):
    """
    Constructs the tool catalog for the Native Tool Calling Agent:
    1. Document_Search: RAG retriever with top-4 chunks (k=4) & fallback.
    2. Topic_Modeling: BERTopic theme discovery.
    3. Sentiment_Analysis: TextBlob polarity & subjectivity.
    4. Info_Extraction: spaCy token-level Named Entity Recognition.
    5. Translate: deep-translator multilingual engine.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def run_doc_search(query: str) -> str:
        clean_q = query.strip().strip('"').strip("'")
        docs = retriever.invoke(clean_q)
        if not docs:
            return "No directly matching excerpt found in the document."
        return "\n\n".join([f"[Excerpt {i+1}]: {d.page_content}" for i, d in enumerate(docs)])

    def run_topic_modeling(text: str) -> str:
        try:
            clean_text = text.strip()
            sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', clean_text) if len(s.strip()) > 15]
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
    llm = ChatOllama(
        model="llama3",
        temperature=0.0
    )

    tools = build_agent_tools(vectorstore)

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

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

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
