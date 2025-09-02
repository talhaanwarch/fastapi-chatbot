"""Chat service for managing conversation flow and message handling."""

import logging
import time
from typing import List, Dict, Any, AsyncGenerator
from fastapi import WebSocket
from .llm_service import LLMService
from .vector_service import VectorService

logger = logging.getLogger(__name__)


class ChatService:
    """Service for handling chat conversations and message processing."""
    
    def __init__(self):
        """Initialize chat service with LLM and vector services."""
        self.llm_service = LLMService()
        self.vector_service = VectorService()
        logger.info("Chat service initialized successfully")
    
    @staticmethod
    def format_messages(chat_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert chat history to standard message format.
        
        Args:
            chat_history: List of chat entries with user/assistant keys
            
        Returns:
            List[Dict[str, str]]: Formatted messages with role and content
        """
        messages = []
        for entry in chat_history:
            if "user" in entry:
                messages.append({"role": "user", "content": entry["user"]})
            elif "assistant" in entry:
                messages.append({"role": "assistant", "content": entry["assistant"]})
        return messages
    
    @staticmethod
    def messages_to_string(messages: List[Dict[str, str]]) -> str:
        """
        Convert messages to string format for processing.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            str: String representation of messages
        """
        messages_str = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            messages_str += f"{role}: {content}\n"
        return messages_str
    
    async def process_message(
        self, 
        user_input: str, 
        chat_history: List[Dict[str, str]], 
        websocket: WebSocket
    ) -> str:
        """
        Process a user message and generate a streaming response.
        
        Args:
            user_input: The user's input message
            chat_history: List of previous chat messages
            websocket: WebSocket connection for streaming response
            
        Returns:
            str: Complete response text for storing in chat history
        """
        total_start = time.time()
        logger.info(f"Processing user message: {user_input}")
        
        # Add user message to history
        chat_history.append({"user": user_input})
        
        # Convert to standard message format
        messages = self.format_messages(chat_history)
        logger.info(f"Chat history contains {len(messages)} messages")
        
        # Step 1: Query refinement (if conversation has history)
        start_time = time.time()
        if len(messages) > 1:
            messages_str = self.messages_to_string(messages)
            query = self.llm_service.refine_query(messages_str, user_input)
        else:
            query = user_input
        
        end_time = time.time()
        logger.info(f"Query refinement completed in {end_time - start_time:.4f} seconds")
        
        # Step 2: Vector search and reranking
        start_time = time.time()
        context = self.vector_service.search_and_rerank(query)
        end_time = time.time()
        logger.info(f"Vector search and reranking completed in {end_time - start_time:.4f} seconds")
        
        # Step 3: Generate streaming response
        start_time = time.time()
        response_text = ""
        
        async for chunk in self._stream_response(messages, context, websocket):
            response_text += chunk
        
        # Send end marker
        await websocket.send_text('[END]')
        
        end_time = time.time()
        logger.info(f"Response generation completed in {end_time - start_time:.4f} seconds")
        
        # Add assistant response to history
        chat_history.append({"assistant": response_text})
        
        total_end = time.time()
        logger.info(f"Total message processing time: {total_end - total_start:.4f} seconds")
        
        return response_text
    
    async def _stream_response(
        self, 
        messages: List[Dict[str, str]], 
        context: str, 
        websocket: WebSocket
    ) -> AsyncGenerator[str, None]:
        """
        Stream response chunks to the WebSocket.
        
        Args:
            messages: Conversation messages
            context: Retrieved context for answering
            websocket: WebSocket connection
            
        Yields:
            str: Response chunks
        """
        try:
            for chunk in self.llm_service.call_stream(messages, context):
                await websocket.send_text(chunk)
                yield chunk
        except Exception as e:
            logger.error(f"Error during response streaming: {e}")
            error_msg = "An error occurred while generating the response. Please try again."
            await websocket.send_text(error_msg)
            yield error_msg