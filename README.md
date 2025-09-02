# FastAPI RAG Chatbot

A production-ready FastAPI application that implements a Retrieval-Augmented Generation (RAG) chatbot using vector search, document reranking, and streaming LLM responses.

## Features

- **Real-time Chat**: WebSocket-based chat interface with streaming responses
- **RAG Architecture**: Combines vector search with language models for accurate, context-aware responses
- **Vector Search**: Uses Qdrant vector database with OpenAI embeddings for document similarity search
- **Document Reranking**: Cohere rerank API for improved relevance of retrieved documents
- **Query Refinement**: Context-aware query refinement based on conversation history
- **Modular Architecture**: Clean, maintainable code structure with proper separation of concerns

## Architecture

### Core Components

- **FastAPI App** (`main.py`): HTTP server and WebSocket endpoints
- **Chat Service** (`chat_service.py`): Orchestrates the conversation flow
- **LLM Service** (`llm_service.py`): Handles language model interactions
- **Vector Service** (`vector_service.py`): Manages vector search and document reranking
- **Configuration** (`config.py`): Centralized configuration management

### Chat Flow

1. **User Input**: Received via WebSocket connection
2. **Query Refinement**: Uses conversation history to refine the query (for multi-turn conversations)
3. **Vector Search**: Retrieves relevant documents from Qdrant vector store
4. **Document Reranking**: Reorders documents by relevance using Cohere
5. **Response Generation**: Streams LLM response with retrieved context
6. **History Management**: Maintains conversation history for context

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd fastapi-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

## Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENROUTER_KEY=your_openrouter_api_key

# Qdrant Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Cohere Reranking
COHERE_API_KEY=your_cohere_api_key

# Langfuse (LLM Observability)
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
```

## Usage

### Development

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# Using the start script
./start.sh

# Or directly with uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t fastapi-chatbot .
docker run -p 8000:8000 --env-file .env fastapi-chatbot
```

## API Endpoints

- **GET /**: Serves the chat interface HTML page
- **WebSocket /**: Real-time chat WebSocket endpoint

## Configuration

The application uses a centralized configuration system in `app/config.py`. Key configuration options:

- **Model Settings**: LLM model selection, temperature, max tokens
- **Vector Search**: Similarity search parameters, collection name
- **Reranking**: Top-N documents to rerank
- **API Endpoints**: Various service URLs and keys

## Project Structure

```
fastapi-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and routes
│   ├── config.py            # Configuration management
│   ├── chat_service.py      # Chat orchestration logic
│   ├── llm_service.py       # Language model interactions
│   ├── vector_service.py    # Vector search and reranking
│   └── utils.py             # Utility functions
├── templates/
│   └── chatting.html        # Chat interface template
├── static/
│   ├── chatting.css         # Chat interface styles
│   └── chatting.js          # Chat interface JavaScript
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose configuration
├── start.sh                 # Production start script
└── README.md               # This file
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **LangChain**: Framework for LLM applications
- **Qdrant**: Vector database for similarity search
- **OpenAI**: Embeddings and language models
- **Cohere**: Document reranking
- **Langfuse**: LLM observability and prompt management
- **WebSockets**: Real-time communication
- **Jinja2**: HTML templating

## Development

### Code Quality

The codebase follows these principles:

- **Modular Design**: Separation of concerns with dedicated service classes
- **Type Hints**: Full type annotation for better IDE support and error detection
- **Documentation**: Comprehensive docstrings for all functions and classes
- **Error Handling**: Proper exception handling with logging
- **Configuration Management**: Centralized configuration with validation

### Adding New Features

1. **Service Layer**: Add business logic to appropriate service classes
2. **Configuration**: Add new settings to `config.py`
3. **Routes**: Add new endpoints to `main.py`
4. **Documentation**: Update docstrings and README

## Monitoring and Observability

The application includes built-in observability through:

- **Logging**: Comprehensive logging throughout the application
- **Langfuse Integration**: LLM call tracking and prompt management
- **Performance Metrics**: Timing information for each processing step
- **Error Tracking**: Detailed error logging and handling

