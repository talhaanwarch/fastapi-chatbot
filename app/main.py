from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Request,
    WebSocket,
    status,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_milvus import Milvus
from utils import call_stream,message_to_str,call_refiner_prompt
from dotenv import load_dotenv
import os
import logging
from humanloop import Humanloop
import time
logging.basicConfig(level=logging.INFO)

# logging.basicConfig(filename='/app/logs/websocket_chat.log',  level=logging.INFO, 
#                     format='%(asctime)s - %(levelname)s - %(message)s')


load_dotenv()
templates = Jinja2Templates(directory="../templates")
app = FastAPI()
app.mount("/static", StaticFiles(directory="../static"), name="static")


client = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"),
)
embeddings = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_KEY"), model_name="jina-embeddings-v3"
)

def create_db_from_file(uploaded_file):
    docs = load_split_file(f"{uploaded_file.filename}")
    vector_store_saved = Milvus.from_documents(
        docs,
        embeddings,
        collection_name="langchain_example",
        connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "secure": True,
        },
    )
    print("Vector store saved")


@app.get("/", response_class=HTMLResponse)
def return_homepage(request: Request):
    return templates.TemplateResponse(request=request, name="chatting.html")


vector_store_loaded = Milvus(
    embeddings,
    connection_args={
        "uri": os.getenv("MILVUS_URI"),
        "token": os.getenv("MILVUS_TOKEN"),
        "secure": True,
    },
    collection_name="UNCITRAL",
)

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

            # Refine the query
            start_time = time.time()
            total_start = time.time()
            messages_str = message_to_str(messages)
            query = call_refiner_prompt(client, messages_str, user_input)
            end_time = time.time()
            logging.info(f"Time taken to refine query: {end_time - start_time:.4f} seconds")

            # Perform similarity search
            start_time = time.time()
            results = vector_store_loaded.similarity_search(query, k=5)
            page_content = "\n".join([i.page_content for i in results])
            end_time = time.time()
            logging.info(f"Time taken to perform similarity search: {end_time - start_time:.4f} seconds")

            # Call stream and send response
            response_text = ''
            start_time = time.time()
            for chunk in call_stream(client, messages, page_content):
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