from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, requests, json
from dotenv import load_dotenv
load_dotenv()
import logging
import time

logging.basicConfig(level=logging.INFO)

def call_stream(messages, context):
    url = "https://api.humanloop.com/v5/prompts/call"
    headers = {
        "X-API-KEY": os.getenv("HUMANLOOP_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "environment": "production",
        "id": "pr_zPxbzZ6eDSnN97v2uMQsj",
        "messages": messages,
        "inputs": {"chunks": context},
        "source": "test",
        "save": True,
        "num_samples": 1,
        "return_inputs": False,
        "stream": True
    }
    
    response = requests.post(url, headers=headers, json=data, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
        
    for line in response.iter_lines():
        if line:
            # Remove the 'data: ' prefix and decode the line
            json_str = line.decode('utf-8').replace('data: ', '')
            # Parse the JSON string
            data = json.loads(json_str)
            res = data['output']
            yield res


def call_refiner_prompt(messages, question):
    url = "https://api.humanloop.com/v5/prompts/call"
    headers = {
        "X-API-KEY": os.getenv("HUMANLOOP_KEY"),
        "Content-Type": "application/json"
    }
    data = {
        "environment": "production",
        "id": "pr_0OIl9UUwDEMHXctJ93OyC",
        "inputs": {"conversation": messages, "question": question},
        "source": "test",
        "save": True,
        "num_samples": 1,
        "return_inputs": False,
        "stream": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    
    return response.json()["logs"][0]["output"]

def jina_rerank( query, docs, top_n=5):

    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('JINA_KEY')}",
    }
    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": top_n,
        "documents": docs,
    }

    response = requests.post(url, headers=headers, json=data)
    response = response.json()["results"]
    index = [i["index"] for i in response]
    docs = [docs[i] for i in index]
    return docs

def message_to_str(messages):
    messages_str = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        messages_str += f"{role}: {content}\n"
    return messages_str