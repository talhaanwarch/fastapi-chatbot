"""FastAPI chatbot application with RAG capabilities."""

import os
import logging
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from .config import config
from .agent_service import AgentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate configuration
config.validate()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A FastAPI application with RAG capabilities using vector search and LLM",
    version="1.0.0"
)

# Setup templates and static files
templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static'))

logger.info(f"Templates directory: {templates_dir}")
logger.info(f"Static directory: {static_dir}")

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Initialize agent service
agent_service = AgentService()

@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request) -> HTMLResponse:
    """
    Serve the chat interface homepage.
    
    Args:
        request: FastAPI request object
        
    Returns:
        HTMLResponse: The chat interface HTML page
    """
    return templates.TemplateResponse(request=request, name="chatting.html")


@app.websocket("/")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    Handle WebSocket connections for real-time chat.
    
    The chat flow uses OpenAI Agents framework with tools:
    1. Query refinement based on conversation history
    2. Vector similarity search for relevant context
    3. Document reranking for better relevance
    4. Streaming agent response generation
    
    Args:
        websocket: WebSocket connection instance
    """
    await websocket.accept()
    chat_history = []  # Store conversation history
    
    try:
        while True:
            # Receive user message
            user_input = await websocket.receive_text()
            logger.info(f"Received user message: {user_input}")
            
            # Process message and stream response using agent
            await agent_service.process_message(user_input, chat_history, websocket)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error occurred: {e}")
        try:
            await websocket.send_text("An error occurred. Please try again.")
        except:
            pass  # Connection might be closed