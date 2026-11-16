"""
Agentic Workflow & ReAct Agent Setup
Capstone Project - Natural Language Processing
Supports Local Ollama LLM (NVIDIA RTX 3050 GPU Accelerated) & Cloud OpenAI
"""

import os
import json
import urllib.request
from typing import Optional, List, Dict, Any

try:
    from langchain.agents import AgentExecutor, create_react_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_react_agent

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# Local Ollama & Cloud OpenAI Chat Models
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from nlp_utils import analyze_sentiment, translate_text


def get_available_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    Queries the local Ollama server to list currently installed models.
    """
    try:
        url = f"{base_url.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            models = [m.get("name") for m in data.get("models", []) if m.get("name")]
            return models if models else ["llama3.2:latest"]
    except Exception:
        return ["llama3.2:latest", "llama3:latest", "mistral:latest", "qwen2.5:3b"]


def build_document_retriever_tool(vectorstore) -> Tool:
    """
    Creates a retrieval tool for the indexed PDF document.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def run_retrieval(query: str) -> str:
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No directly matching content found in the uploaded document."
            
            formatted_results = []
            for i, doc in enumerate(docs, 1):
                clean_snippet = doc.page_content.replace("\n", " ").strip()
                formatted_results.append(f"[Excerpt {i}]: {clean_snippet}")
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error querying local document vectorstore: {str(e)}"

    return Tool(
        name="Document_Search",
        func=run_retrieval,
        description=(
            "Use this tool to search information, facts, definitions, and data "
            "inside the user's uploaded research paper or PDF document. "
            "Input should be a concise semantic search query."
        )
    )


def build_sentiment_tool() -> Tool:
    """
    Creates a Sentiment Analysis tool using TextBlob.
    """
    def run_sentiment(text: str) -> str:
        res = analyze_sentiment(text)
        return (
            f"Sentiment: {res['sentiment']} | "
            f"Polarity Score: {res['polarity']} (-1.0 to +1.0) | "
            f"Subjectivity Score: {res['subjectivity']} (0.0 to 1.0)"
        )

    return Tool(
        name="Sentiment_Analysis",
        func=run_sentiment,
        description=(
            "Analyzes the emotional tone, polarity, and subjectivity of any given statement or passage. "
            "Input should be the raw text snippet to analyze."
        )
    )


def build_translation_tool() -> Tool:
    """
    Creates a Multilingual Translation tool using deep-translator.
    Format: 'target_lang:text' or simply 'text' (defaulting to English).
    """
    def run_translation(query: str) -> str:
        target_lang = "en"
        text_to_translate = query

        # Parse custom target language syntax if provided, e.g. "es:Hello world" or "fr|Hello"
        if ":" in query and len(query.split(":", 1)[0].strip()) <= 5:
            target_lang, text_to_translate = query.split(":", 1)
            target_lang = target_lang.strip().lower()
        elif "|" in query and len(query.split("|", 1)[0].strip()) <= 5:
            target_lang, text_to_translate = query.split("|", 1)
            target_lang = target_lang.strip().lower()

        translated = translate_text(text_to_translate.strip(), target_lang=target_lang)
        return f"[Target: {target_lang}] Translated Text: {translated}"

    return Tool(
        name="Translation",
        func=run_translation,
        description=(
            "Translates text across languages. By default translates to English ('en'). "
            "To specify a target language, prefix input with language code and colon, "
            "e.g., 'es:This is a research summary' for Spanish or 'hi:Hello' for Hindi."
        )
    )


def get_react_prompt() -> PromptTemplate:
    """
    Prompt Engineering: Returns a robust, calibrated ReAct prompt template designed
    to work seamlessly across both local open-weights LLMs (like LLaMA 3.2 on RTX 3050)
    and cloud LLMs.
    """
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

IMPORTANT RULES:
1. When you receive an Observation that answers the Question, do NOT invoke any more tools. You MUST write 'Thought: I now know the final answer' followed by 'Final Answer: ...'.
2. Never hallucinate or invent tool names outside of [{tool_names}].
3. For general greetings or simple questions, you can respond directly with 'Thought: I now know the final answer' and provide 'Final Answer: ...'.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    return PromptTemplate.from_template(template)


def create_agent(
    vectorstore,
    model_provider: str = "ollama",
    model_name: str = "llama3.2",
    ollama_base_url: str = "http://localhost:11434",
    openai_api_key: Optional[str] = None
) -> AgentExecutor:
    """
    Initializes and returns a LangChain ReAct AgentExecutor equipped with:
    1. Document_Search (RAG retriever from Chroma)
    2. Web_Search (DuckDuckGo search for live external knowledge)
    3. Sentiment_Analysis (TextBlob tone and subjectivity)
    4. Translation (deep-translator multilingual capabilities)

    Args:
        vectorstore: Chroma vector store instance populated with PDF embeddings.
        model_provider: 'ollama' (Local RTX 3050 GPU accelerated) or 'openai' (Cloud).
        model_name: Model identifier (e.g. 'llama3.2', 'llama3:latest', 'gpt-3.5-turbo').
        ollama_base_url: Local Ollama service endpoint (default: http://localhost:11434).
        openai_api_key: OpenAI API key (required only if model_provider='openai').

    Returns:
        AgentExecutor: Ready-to-invoke ReAct agent instance.
    """
    # 1. Initialize Generative AI Model
    if model_provider.lower() == "ollama":
        # Local Ollama with GPU acceleration (RTX 3050 6GB)
        llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.0
        )
    else:
        # Cloud OpenAI fallback
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API Key is missing. Please set the OPENAI_API_KEY environment variable "
                "or switch to 'Local Ollama' mode in the application sidebar."
            )
        llm = ChatOpenAI(
            model=model_name or "gpt-3.5-turbo",
            temperature=0.0,
            api_key=api_key
        )

    # 2. Assemble Multimodal Tools
    document_search_tool = build_document_retriever_tool(vectorstore)
    web_search_tool = Tool(
        name="Web_Search",
        func=DuckDuckGoSearchRun().run,
        description=(
            "Useful when you need current information from the internet, academic citations, "
            "recent developments, or facts not present in the uploaded research document."
        )
    )
    sentiment_tool = build_sentiment_tool()
    translation_tool = build_translation_tool()

    tools = [
        document_search_tool,
        web_search_tool,
        sentiment_tool,
        translation_tool
    ]

    # 3. ReAct Prompt Engineering
    prompt = get_react_prompt()

    # 4. Construct ReAct Agent
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 5. Return Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=6
    )

    return agent_executor
