"""Agent tools for RAG operations using OpenAI Agents framework."""

import logging
from typing import List, Optional
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
        logger.info(f"Performing vector search for query: {query[:100]}...")
        vector_service = get_vector_service()
        documents = vector_service.similarity_search(query)
        result = "\n--------------------------------------------------\n".join(documents)
        logger.info(f"Vector search completed, returned {len(documents)} documents")
        return result
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
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
        logger.info(f"Refining query: {question[:100]}...")
        llm_service = get_llm_service()
        refined = llm_service.refine_query(conversation, question)
        logger.info(f"Query refined from '{question}' to '{refined}'")
        return refined
    except Exception as e:
        logger.error(f"Query refinement failed: {e}")
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
        logger.info(f"Reranking documents for query: {query[:100]}...")
        
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
        logger.info(f"Document reranking completed, processed {len(documents)} documents")
        return result
        
    except Exception as e:
        logger.error(f"Document reranking failed: {e}")
        return documents_str  # Return original if reranking fails