"""Agent tools for RAG operations using OpenAI Agents framework."""

import logging
import time
from typing import Optional
from agents import function_tool

logger = logging.getLogger(__name__)

# Lazy-loaded services to avoid connection errors during import
_vector_service: Optional['VectorService'] = None
_llm_service: Optional['LLMService'] = None


def get_vector_service():
    """Get or create vector service instance."""
    global _vector_service
    if _vector_service is None:
        from .vector_service import VectorService
        _vector_service = VectorService()
    return _vector_service


def get_llm_service():
    """Get or create LLM service instance."""
    global _llm_service
    if _llm_service is None:
        from .llm_service import LLMService
        _llm_service = LLMService()
    return _llm_service


@function_tool
def vector_search(query: str) -> str:
    """
    Perform vector similarity search to find relevant documents.
    
    Args:
        query: The search query to find relevant documents
        
    Returns:
        str: Retrieved documents separated by delimiters (without reranking)
    """
    try:
        start_time = time.time()
        logger.info(f"tool=vector_search status=start query_preview='{query[:100]}'")
        vector_service = get_vector_service()
        documents = vector_service.similarity_search(query)
        result = "\n--------------------------------------------------\n".join(documents)
        elapsed = time.time() - start_time
        logger.info(f"tool=vector_search status=success docs={len(documents)} elapsed_sec={elapsed:.4f}")
        return result
    except Exception as e:
        logger.exception(f"tool=vector_search status=error message='{str(e)}'")
        return "No relevant documents found due to search error."


@function_tool
def refine_query(conversation: str, question: str) -> str:
    """
    Refine the user's query based on conversation context.
    
    Args:
        conversation: The conversation history as a string
        question: The current user question
        
    Returns:
        str: Refined query that better captures user intent
    """
    try:
        start_time = time.time()
        logger.info(f"tool=refine_query status=start question_preview='{question[:100]}' conv_chars={len(conversation)}")
        llm_service = get_llm_service()
        refined = llm_service.refine_query(conversation, question)
        elapsed = time.time() - start_time
        logger.info(f"tool=refine_query status=success original_preview='{question[:80]}' refined_preview='{refined[:80]}' elapsed_sec={elapsed:.4f}")
        return refined
    except Exception as e:
        logger.exception(f"tool=refine_query status=error message='{str(e)}'")
        return question


@function_tool
def rerank_documents(query: str, documents_str: str) -> str:
    """
    Rerank documents using Cohere's rerank API for better relevance.
    
    Args:
        query: The original search query
        documents_str: Documents separated by delimiters
        
    Returns:
        str: Reranked documents in order of relevance
    """
    try:
        start_time = time.time()
        logger.info(f"tool=rerank_documents status=start query_preview='{query[:100]}'")
        
        # Split documents by delimiter
        documents = documents_str.split("\n--------------------------------------------------\n")
        
        # Filter out empty documents
        documents = [doc.strip() for doc in documents if doc.strip()]
        
        if not documents:
            return "No documents to rerank."
            
        # Use vector service reranking
        vector_service = get_vector_service()
        reranked = vector_service.rerank_documents(query, documents)
        
        # Join back with delimiter
        result = "\n--------------------------------------------------\n".join(reranked)
        elapsed = time.time() - start_time
        logger.info(f"tool=rerank_documents status=success input_docs={len(documents)} output_docs={len(reranked)} elapsed_sec={elapsed:.4f}")
        return result
        
    except Exception as e:
        logger.exception(f"tool=rerank_documents status=error message='{str(e)}'")
        return documents_str  # Return original if reranking fails