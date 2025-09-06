"""Agent service for handling chat conversations using OpenAI Agents framework."""

import logging
import time
from typing import AsyncGenerator
from fastapi import WebSocket
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
from . import agent_tools
from .config import config

logger = logging.getLogger(__name__)


class AgentService:
    """Service for handling chat conversations using OpenAI Agents."""
    
    def __init__(self):
        """Initialize the agent service with configured agent."""
        
        # Hardcoded instructions (replacing Langfuse prompts temporarily)
        instructions = """You are a helpful RAG (Retrieval-Augmented Generation) assistant that can search through documents to answer questions.

When a user asks a question:
1. If there's conversation history, use refine_query to better understand what the user is asking based on context
2. Use vector_search to find relevant documents that might contain the answer
3. For better results, use rerank_documents to reorder the retrieved documents by relevance
4. Use the retrieved and reranked information to provide a comprehensive, accurate answer
5. If you can't find relevant information, say so clearly

Always be helpful, accurate, and cite the information you find in the documents when possible.
"""
        
        # Create model object for the agent
        chat_model = config.get_chat_model()
        
        self.agent = Agent(
            name="RAG Assistant",
            instructions=instructions,
            tools=[agent_tools.vector_search, agent_tools.refine_query, agent_tools.rerank_documents],
            model=chat_model
        )
        
        logger.info("Agent service initialized successfully")
    
    async def process_message(
        self, 
        user_input: str, 
        chat_history: list, 
        websocket: WebSocket
    ) -> str:
        """
        Process a user message using the agent and stream the response.
        
        Args:
            user_input: The user's input message
            chat_history: List of previous chat messages
            websocket: WebSocket connection for streaming response
            
        Returns:
            str: Complete response text for storing in chat history
        """
        total_start = time.time()
        logger.info(f"Processing user message with agent: {user_input}")
        
        # Add user message to history
        chat_history.append({"user": user_input})
        
        # Convert history to conversation context string for the agent
        conversation_context = self._format_conversation_context(chat_history[:-1])  # Exclude current message
        
        # Create input message with context
        if conversation_context:
            agent_input = f"Conversation context:\n{conversation_context}\n\nCurrent question: {user_input}"
        else:
            agent_input = user_input
        
        # Stream response using the agent
        start_time = time.time()
        response_text = ""
        
        try:
            async for chunk in self._stream_agent_response(agent_input, websocket):
                response_text += chunk
            
            # Send end marker
            await websocket.send_text('[END]')
            
        except Exception as e:
            logger.error(f"Agent processing failed: {e}")
            error_msg = "An error occurred while generating the response. Please try again."
            await websocket.send_text(error_msg)
            await websocket.send_text('[END]')
            response_text = error_msg
        
        end_time = time.time()
        logger.info(f"Agent response generation completed in {end_time - start_time:.4f} seconds")
        
        # Add assistant response to history
        chat_history.append({"assistant": response_text})
        
        total_end = time.time()
        logger.info(f"Total agent message processing time: {total_end - total_start:.4f} seconds")
        
        return response_text
    
    def _format_conversation_context(self, chat_history: list) -> str:
        """
        Format chat history into a conversation context string.
        
        Args:
            chat_history: List of chat entries
            
        Returns:
            str: Formatted conversation context
        """
        if not chat_history:
            return ""
            
        context_parts = []
        for entry in chat_history:
            if "user" in entry:
                context_parts.append(f"User: {entry['user']}")
            elif "assistant" in entry:
                context_parts.append(f"Assistant: {entry['assistant']}")
        
        return "\n".join(context_parts)
    
    async def _stream_agent_response(
        self, 
        agent_input: str, 
        websocket: WebSocket
    ) -> AsyncGenerator[str, None]:
        """
        Stream agent response chunks to the WebSocket.
        
        Args:
            agent_input: Input message for the agent
            websocket: WebSocket connection
            
        Yields:
            str: Response chunks
        """
        try:
            # Run the agent with streaming
            result = Runner.run_streamed(self.agent, input=agent_input)
            
            # Stream events from the agent
            async for event in result.stream_events():
                # Handle raw response events (token-by-token streaming)
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    chunk = event.data.delta
                    if chunk:
                        await websocket.send_text(chunk)
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Error during agent response streaming: {e}")
            error_msg = "An error occurred while generating the response. Please try again."
            await websocket.send_text(error_msg)
            yield error_msg