"""Language model service for handling LLM interactions."""

import logging
from openai import OpenAI
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
        
        # Prompt for query refinement
        self.refiner_prompt_template = """Given the following conversation history and a new question, reformulate the question to be more specific and standalone, incorporating relevant context from the conversation history.

Conversation history:
{conversation}

New question: {question}

Reformulated question:"""
        
        logger.info("LLM service initialized successfully")
    
    def refine_query(self, messages_str: str, question: str) -> str:
        """
        Refine the user's query based on conversation history.
        
        Args:
            messages_str: String representation of conversation history
            question: Current user question
            
        Returns:
            str: Refined query or original question if refinement fails
        """
        prompt = self.refiner_prompt_template.format(conversation=messages_str, question=question)
        
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