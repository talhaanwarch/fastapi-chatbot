"""Configuration management for the FastAPI chatbot application."""

import os
from typing import Optional
from dotenv import load_dotenv
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENROUTER_KEY: Optional[str] = os.getenv("OPENROUTER_KEY")
    
    # Qdrant Configuration
    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = "uncitral"
    
    # Cohere Configuration
    COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "google/gemini-2.0-flash-001"
    RERANK_MODEL: str = "rerank-v3.5"
    
    # API Configuration
    CHAT_TEMPERATURE: float = 0.1
    CHAT_MAX_TOKENS: int = 2000
    REFINER_MAX_TOKENS: int = 200
    
    # Vector Search Configuration
    SIMILARITY_SEARCH_K: int = 10
    RERANK_TOP_N: int = 6
    
    @classmethod
    def get_chat_model(cls) -> OpenAIChatCompletionsModel:
        """
        Create and return a configured chat model object.
        
        Returns:
            OpenAIChatCompletionsModel: Configured model for chat operations
        """
        # Create OpenAI client with OpenRouter configuration
        openai_client = AsyncOpenAI(
            api_key=cls.OPENROUTER_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        return OpenAIChatCompletionsModel(
            model=cls.CHAT_MODEL,
            openai_client=openai_client
        )
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required configuration is present."""
        required_keys = [
            "OPENAI_API_KEY", "OPENROUTER_KEY", "QDRANT_URL", 
            "QDRANT_API_KEY", "COHERE_API_KEY"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")


# Global configuration instance
config = Config()