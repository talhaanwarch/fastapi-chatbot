from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, requests
from dotenv import load_dotenv
load_dotenv()


def call_stream(client, messages,context):
    response = client.prompts.call_stream(
        environment="production",
        id="pr_XE7txDMTqX6DR8Fk40kWd",
        messages=messages,
        inputs={"chunks": context},
        source="test",
        save=True,
        num_samples=1,
        return_inputs=False,
        
    )
    for chunk in response:
        yield chunk.output_message.content


def message_to_str(messages):
    messages_str = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        messages_str += f"{role}: {content}\n"
    return messages_str

def call_refiner_prompt(client, messages,question):
    response = client.prompts.call(
        environment="production",
        id="pr_VR2yMxzTxLxjy1zGpo6Sb",
        inputs={"conversation": messages,"question":question},
        source="test",
        save=True,
        num_samples=1,
        return_inputs=False,
        
    )
    return response.logs[0].output

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
    print("Length of docs 1",len(docs))
    index = [i["index"] for i in response]
    print('index',index)
    docs = [docs[i] for i in index]
    print("Length of docs 2",len(docs))
    return docs