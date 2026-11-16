"""
NLP Utility Functions for Agentic RAG-Based AI Research Assistant
Capstone Project - Natural Language Processing
"""

import os
import re
from typing import List, Dict, Any, Union
import pandas as pd
from pypdf import PdfReader
from textblob import TextBlob
from deep_translator import GoogleTranslator
from bertopic import BERTopic

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def extract_text_from_pdf(pdf_source: Union[str, Any]) -> str:
    """
    Extracts text from a PDF file path or file-like object (e.g. Streamlit UploadedFile).
    
    Args:
        pdf_source: File path (str) or file-like binary buffer (UploadedFile, BytesIO).
        
    Returns:
        Extracted and sanitized text as a single string.
    """
    try:
        reader = PdfReader(pdf_source)
        extracted_pages = []
        
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                # Clean extraneous form feeds and repeated whitespace
                cleaned_page = re.sub(r'\s+', ' ', page_text).strip()
                extracted_pages.append(cleaned_page)
                
        full_text = "\n\n".join(extracted_pages)
        if not full_text.strip():
            raise ValueError("No extractable text found in the provided PDF file.")
            
        return full_text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")


def process_and_vectorize(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200, 
    persist_directory: str = None
) -> Chroma:
    """
    Splits text into chunks using RecursiveCharacterTextSplitter,
    generates embeddings with HuggingFace ('all-MiniLM-L6-v2'),
    and stores them in a Chroma vector store.
    
    Args:
        text: Raw text to be chunked and indexed.
        chunk_size: Maximum character length per text chunk.
        chunk_overlap: Overlap between consecutive chunks to preserve context.
        persist_directory: Optional path to save Chroma vectorstore locally on disk.
        
    Returns:
        Chroma vectorstore instance ready for retrieval.
    """
    if not text or not text.strip():
        raise ValueError("Cannot vectorize empty text.")

    # 1. Text Preprocessing & Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.create_documents([text])
    
    # 2. Vectorization using HuggingFace sentence-transformers
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # 3. Store into Chroma Vectorstore
    if persist_directory:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        
    return vectorstore


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Performs sentiment and subjectivity analysis using TextBlob.
    
    Args:
        text: Input text string to analyze.
        
    Returns:
        Dictionary containing polarity (-1.0 to 1.0), subjectivity (0.0 to 1.0),
        and categorical sentiment label.
    """
    if not text or not text.strip():
        return {
            "polarity": 0.0,
            "subjectivity": 0.0,
            "sentiment": "Neutral",
            "summary": "Empty text provided."
        }
        
    blob = TextBlob(text)
    polarity = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)
    
    if polarity > 0.1:
        sentiment = "Positive"
    elif polarity < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "sentiment": sentiment,
        "summary": f"Sentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"
    }


def get_topic_modeling(texts: Union[str, List[str]], top_n_topics: int = 5) -> Dict[str, Any]:
    """
    Discovers latent topics and semantic clusters from texts using BERTopic.
    
    Args:
        texts: A string (split by sentences/paragraphs) or list of text strings.
        top_n_topics: Number of top topics to summarize.
        
    Returns:
        Dictionary with topic breakdown, keywords, and formatted summary table.
    """
    # Prepare documents list
    if isinstance(texts, str):
        # Split into substantial sentences or paragraphs
        raw_docs = [p.strip() for p in texts.split("\n") if len(p.strip()) > 30]
        if len(raw_docs) < 5:
            raw_docs = [s.strip() for s in re.split(r'(?<=[.?!])\s+', texts) if len(s.strip()) > 20]
    else:
        raw_docs = [t.strip() for t in texts if len(t.strip()) > 20]
        
    if len(raw_docs) < 3:
        return {
            "status": "warning",
            "message": "Insufficient text chunks for meaningful statistical topic modeling (requires at least 3 distinct passages).",
            "topics": []
        }
        
    try:
        # Initialize BERTopic with lightweight embedding model and low min_topic_size for small-scale docs
        topic_model = BERTopic(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=max(2, len(raw_docs) // 10),
            verbose=False
        )
        topics, probs = topic_model.fit_transform(raw_docs)
        
        topic_info = topic_model.get_topic_info()
        formatted_topics = []
        
        # Collect top topics (excluding outlier topic -1 if others exist)
        for _, row in topic_info.head(top_n_topics + 1).iterrows():
            topic_id = row['Topic']
            name = row.get('Name', f"Topic {topic_id}")
            count = row.get('Count', 0)
            representation = topic_model.get_topic(topic_id)
            keywords = [word for word, score in representation[:5]] if representation else []
            
            formatted_topics.append({
                "topic_id": int(topic_id),
                "name": name,
                "document_count": int(count),
                "keywords": keywords
            })
            
        return {
            "status": "success",
            "total_documents": len(raw_docs),
            "topics": formatted_topics,
            "topic_info_df": topic_info.to_dict(orient="records")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"BERTopic modeling could not be completed: {str(e)}",
            "topics": []
        }


def translate_text(text: str, target_lang: str = 'en') -> str:
    """
    Translates text to the specified target language using deep-translator (GoogleTranslator).
    Handles long text by breaking into chunks to stay within API length boundaries.
    
    Args:
        text: Input string in any supported source language.
        target_lang: ISO language code (e.g. 'en', 'es', 'fr', 'de', 'hi', 'zh-CN').
        
    Returns:
        Translated string.
    """
    if not text or not text.strip():
        return ""
        
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        
        # Deep-translator chunk limit is ~5000 characters
        max_chunk = 4000
        if len(text) <= max_chunk:
            return translator.translate(text)
            
        chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
        translated_chunks = [translator.translate(chunk) for chunk in chunks]
        return " ".join(translated_chunks)
    except Exception as e:
        return f"Translation error: {str(e)}"
