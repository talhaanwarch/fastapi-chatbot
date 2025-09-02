"""Vector store service for similarity search and document retrieval."""

import logging
from typing import List
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
import cohere
from .config import config

logger = logging.getLogger(__name__)


class VectorService:
    """Service for handling vector operations and document retrieval."""
    
    def __init__(self):
        """Initialize the vector service with Qdrant client and embeddings."""
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
        
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENAI_API_KEY,
        )
        
        logger.info("Initializing Qdrant vector store...")
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=config.QDRANT_COLLECTION_NAME,
            embedding=self.embeddings,
        )
        logger.info("Vector store initialized successfully")
        
        # Initialize Cohere client for reranking
        self.cohere_client = cohere.ClientV2(api_key=config.COHERE_API_KEY)
    
    def similarity_search(self, query: str, k: int = None) -> List[str]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (defaults to config value)
            
        Returns:
            List[str]: List of retrieved document contents
        """
        if k is None:
            k = config.SIMILARITY_SEARCH_K
            
        try:
            results = self.vector_store.similarity_search(query, k=k)
            documents = [doc.page_content for doc in results]
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:100]}...")
            return documents
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def rerank_documents(self, query: str, documents: List[str], top_n: int = None) -> List[str]:
        """
        Rerank documents using Cohere's rerank API.
        
        Args:
            query: Original search query
            documents: List of documents to rerank
            top_n: Number of top documents to return (defaults to config value)
            
        Returns:
            List[str]: Reranked documents in order of relevance
        """
        if not documents:
            return documents
            
        if top_n is None:
            top_n = config.RERANK_TOP_N
        
        try:
            results = self.cohere_client.rerank(
                model=config.RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            
            # Extract reranked documents in order of relevance
            reranked_docs = [documents[result.index] for result in results.results]
            logger.info(f"Reranked {len(documents)} documents, returning top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Document reranking failed: {e}")
            return documents[:top_n]  # Fallback to original order
    
    def search_and_rerank(self, query: str, k: int = None, top_n: int = None) -> str:
        """
        Perform similarity search followed by reranking.
        
        Args:
            query: Search query
            k: Number of documents to initially retrieve
            top_n: Number of top documents to return after reranking
            
        Returns:
            str: Joined content of reranked documents
        """
        # Get initial documents
        documents = self.similarity_search(query, k)
        
        # Rerank documents
        reranked_docs = self.rerank_documents(query, documents, top_n)
        
        # Join documents with separator
        return "\n--------------------------------------------------\n".join(reranked_docs)