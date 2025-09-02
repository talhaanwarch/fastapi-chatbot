from fastapi import (
    FastAPI,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from .utils import call_stream,message_to_str,call_refiner_prompt,cohere_rerank
from dotenv import load_dotenv
import os
import logging
import time

logging.basicConfig(level=logging.INFO)

load_dotenv()
templates_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
logging.info(f"templates_dir {templates_dir}")
templates = Jinja2Templates(directory=templates_dir)

app = FastAPI()
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../static'))
logging.info(f"static_dir {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)
logging.info("Reading qdrant client start")
vector_store_loaded = QdrantVectorStore(
    client=qdrant_client,
    collection_name="uncitral",
    embedding=embeddings,
)
logging.info("Reading qdrant client end")

@app.get("/", response_class=HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request=request, name="chatting.html")

    
"""
Step1: extract year
Step2: Rewrite Query
Step3: Vector store
Step4: LLM call
Step5: Rerank
"""
@app.websocket("/")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    chat_history = []  # List to store the chat history
    try:
        while True:
            # Receive user input
            user_input = await websocket.receive_text()
            logging.info(f"user message {user_input}")
            # Append the user message to the chat history
            chat_history.append({"user": user_input})

            # Convert chat_history to the correct format and extend messages
            messages = []
            for entry in chat_history:
                if "user" in entry:
                    messages.append({"role": "user", "content": entry["user"]})
                elif "assistant" in entry:
                    messages.append({"role": "assistant", "content": entry["assistant"]})
            logging.info(f"=====messages====={len(messages)}")
            # Refine the query
            start_time = time.time()
            total_start = time.time()
            if len(messages)>1:
                messages_str = message_to_str(messages)
                query = call_refiner_prompt( messages_str, user_input)
                end_time = time.time()
                logging.info(f"Time taken to refine query: {end_time - start_time:.4f} seconds")
            else:
                query = user_input

            # Perform similarity search
            start_time = time.time()
            results = vector_store_loaded.similarity_search(query, k=10)
            result_content =[i.page_content for i in results]
            end_time = time.time()
            logging.info(f"Time taken to perform similarity search: {end_time - start_time:.4f} seconds")

            # Rerank
            start_time = time.time()
            result_content = cohere_rerank(query,result_content)
            page_content = "\n--------------------------------------------------\n".join(result_content)
            end_time = time.time()
            logging.info(f"Time taken to rerank: {end_time - start_time:.4f} seconds")
            

            # Call stream and send response
            response_text = ''
            start_time = time.time()
            for chunk in call_stream(messages, page_content):
                await websocket.send_text(chunk)
                response_text += chunk
            # Send a stop word to indicate the end of the message
            await websocket.send_text('[END]')
            end_time = time.time()
            logging.info(f"Time taken to call stream and send response: {end_time - start_time:.4f} seconds")
            total_end = time.time()
            logging.info(f"Total time taken: {total_end - total_start:.4f} seconds")
            # Append the assistant's response to the chat history
            chat_history.append({"assistant": response_text})

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        await websocket.send_text("An error occurred. Please try again.")