# FastAPI RAG Chatbot

A production-ready FastAPI application that implements an agent-based Retrieval-Augmented Generation (RAG) chatbot using OpenAI Agents framework, vector search, document reranking, and streaming LLM responses.

## Features

- **Agent-Based Architecture**: Uses OpenAI Agents framework with tool calling for intelligent query processing
- **Real-time Chat**: WebSocket-based chat interface with streaming responses
- **RAG Architecture**: Combines vector search with language models for accurate, context-aware responses
- **Vector Search**: Uses Qdrant vector database with OpenAI embeddings for document similarity search
- **Document Reranking**: Cohere rerank API for improved relevance of retrieved documents
- **Query Refinement**: Context-aware query refinement based on conversation history
- **Modular Architecture**: Clean, maintainable code structure with proper separation of concerns

## Architecture

### Core Components

- **FastAPI App** (`main.py`): HTTP server and WebSocket endpoints
- **Agent Service** (`agent_service.py`): Orchestrates conversations using OpenAI Agents framework
- **Agent Tools** (`agent_tools.py`): Function tools for vector search, query refinement, and document reranking
- **LLM Service** (`llm_service.py`): Handles language model interactions for query refinement
- **Vector Service** (`vector_service.py`): Manages vector search and document reranking
- **Configuration** (`config.py`): Centralized configuration management

### Agent-Based Chat Flow

1. **User Input**: Received via WebSocket connection
2. **Agent Processing**: OpenAI Agent processes the query using available tools:
   - **Query Refinement**: Uses conversation history to refine queries for multi-turn conversations
   - **Vector Search**: Retrieves relevant documents from Qdrant vector store
   - **Document Reranking**: Reorders documents by relevance using Cohere
3. **Response Generation**: Agent streams response with retrieved and reranked context
4. **History Management**: Maintains conversation history for context-aware responses

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
# OpenAI Configuration (for embeddings)
OPENAI_API_KEY=your_openai_api_key

# OpenRouter Configuration (for chat models)
OPENROUTER_KEY=your_openrouter_api_key

# Qdrant Vector Database
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Cohere Reranking
COHERE_API_KEY=your_cohere_api_key

# Langfuse (LLM Observability - Optional)
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

- **Model Settings**:
  - `CHAT_MODEL`: Chat model (default: "google/gemini-2.0-flash-001" via OpenRouter)
  - `EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")
  - `CHAT_TEMPERATURE`: Response temperature (default: 0.1)
  - `REFINER_MAX_TOKENS`: Max tokens for query refinement (default: 200)

- **Vector Search**:
  - `SIMILARITY_SEARCH_K`: Number of documents to retrieve (default: 10)
  - `QDRANT_COLLECTION_NAME`: Vector collection name (default: "uncitral")

- **Reranking**:
  - `RERANK_MODEL`: Cohere rerank model (default: "rerank-v3.5")
  - `RERANK_TOP_N`: Top-N documents to return after reranking (default: 6)

- **API Endpoints**: OpenAI, OpenRouter, Qdrant, and Cohere API configurations

## Project Structure

```
fastapi-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and routes
│   ├── config.py            # Configuration management
│   ├── agent_service.py     # Agent orchestration using OpenAI Agents
│   ├── agent_tools.py       # Function tools for RAG operations
│   ├── llm_service.py       # Language model interactions
│   └── vector_service.py    # Vector search and reranking
├── templates/
│   └── chatting.html        # Chat interface template
├── static/
│   ├── chatting.css         # Chat interface styles
│   └── chatting.js          # Chat interface JavaScript
├── prep/
│   └── data/
│       └── extract_pdf_unictral.ipynb  # Data preparation notebook
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose configuration
├── start.sh                 # Production start script
└── README.md               # This file
```

## Dependencies

- **FastAPI**: Web framework for building APIs
- **OpenAI Agents**: Framework for building AI agents with tool calling
- **LangChain**: Framework for LLM applications and vector stores
- **Qdrant**: Vector database for similarity search
- **OpenAI**: Embeddings and language models via OpenRouter
- **Cohere**: Document reranking API
- **Langfuse**: LLM observability and prompt management
- **WebSockets**: Real-time communication
- **Jinja2**: HTML templating
- **Python-dotenv**: Environment variable management

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

