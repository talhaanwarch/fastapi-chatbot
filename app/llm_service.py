"""Language model service for handling LLM interactions."""

import logging
from typing import Generator, Optional
from openai import OpenAI
from langfuse import Langfuse
from .config import config

logger = logging.getLogger(__name__)


class LLMService:
    """Service for handling language model interactions."""
    
    def __init__(self):
        """Initialize the LLM service with clients and prompts."""
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_KEY,
        )
        
        self.langfuse = Langfuse(
            secret_key=config.LANGFUSE_SECRET_KEY,
            public_key=config.LANGFUSE_PUBLIC_KEY,
            host=config.LANGFUSE_HOST
        )
        
        # Load prompts from Langfuse
        self.refiner_prompt = self.langfuse.get_prompt("refiner_prompt")
        self.qa_prompt = self.langfuse.get_prompt("qa_prompt")
        
        logger.info("LLM service initialized successfully")
    
    def call_stream(self, messages: list, context: str) -> Generator[str, None, None]:
        """
        Generate streaming response from the language model.
        
        Args:
            messages: List of conversation messages
            context: Retrieved context for answering questions
            
        Yields:
            str: Streaming response chunks
        """
        user_question = messages[-1]['content'] if messages else ""
        
        # Compile the prompt with context and question
        prompt = self.qa_prompt.compile(chunks=context, question=user_question)
        
        conversation_messages = [
            {"role": "system", "content": prompt}
        ]
        conversation_messages.extend(messages)
        
        try:
            stream = self.client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=conversation_messages,
                stream=True,
                temperature=config.CHAT_TEMPERATURE,
                max_tokens=config.CHAT_MAX_TOKENS
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            yield "Error: Unable to process your request. Please try again."
    
    def refine_query(self, messages_str: str, question: str) -> str:
        """
        Refine the user's query based on conversation history.
        
        Args:
            messages_str: String representation of conversation history
            question: Current user question
            
        Returns:
            str: Refined query or original question if refinement fails
        """
        prompt = self.refiner_prompt.compile(conversation=messages_str, question=question)
        
        try:
            response = self.client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[
                    {"role": "system", "content": prompt}
                ],
                temperature=config.CHAT_TEMPERATURE,
                max_tokens=config.REFINER_MAX_TOKENS
            )
            
            refined_query = response.choices[0].message.content.strip()
            logger.info(f"Query refined from '{question}' to '{refined_query}'")
            return refined_query
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            return question