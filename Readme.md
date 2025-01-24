# FastAPI App for RAG

This project is designed to build a Retriever-Augmented-Generation (RAG) chatbot application using FastAPI and Docker. The chatbot can perform Q&A based on uploaded `.pdf` or `.docx` files.

This project is a forked and inspired from [this repository](https://github.com/jodog0412/rag-chatbot-app-with-fastapi).

![Project Image](https://github.com/user-attachments/assets/e502f8cb-a38f-463d-b918-3a445f3851f8)

## Table of Contents

- Getting Started
- Usage
- Features/Updates

## Getting Started

1. Clone this repository to your local machine:

   ```sh
   git clone https://github.com/talhaanwarch/rag-chatbot-app-with-fastapi.git
   cd langchain-app-with-fastapi
   ```

2. Set the required credentials in the `.env` file located in the `main` directory at the same level as the `app` directory. These providers offer free tiers:
- [Groq](https://groq.com/): For LLM
- [Cohere](https://cohere.com/): For Embeddings.
- [Milvus](https://zilliz.io/): For Vector Database.

   ```sh
   # app/.env
   GROQ_API_KEY=
   COHERE_API_KEY=
   MILVUS_URI=
   MILVUS_TOKEN=
   ```

## Usage
### Prompt
Make sure to modify the prompt in `app/prompt.txt` according to your needs.
### Running the Docker Compose File

To start the local server using Docker Compose, run:

```sh
docker-compose up -d
```
### Upload file
Users can upload `PDF` and `DOCX` files to the `/upload` endpoint. Currently, only one file can be uploaded, and any subsequent uploads will overwrite the previously uploaded file. We're working to improve this feature to allow multiple file uploads.   
## Features/Updates

- [x] Build a history-aware RAG chatbot with minimal dependencies on LangChain.
- [x] Implement a client-server RESTful API.
- [x] Implement WebSocket connections.
## Changs made by me
- [x] Switch from Chroma to Milvus/Zilliz free tier.
- [x] Use GROQ Llama and Cohere embeddings, as both are free.
- [x] Create a Docker Compose file to run the application in a containerized environment.
- [ ] Test the application on the cloud.
