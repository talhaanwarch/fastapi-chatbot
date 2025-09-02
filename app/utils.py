from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, requests, json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
import logging
import cohere
from langfuse import Langfuse

logging.basicConfig(level=logging.INFO)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"),
)
langfuse = Langfuse(
  secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
  public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
  host="https://us.cloud.langfuse.com"
)
REFINER_SYSTEM_PROMPT = langfuse.get_prompt("refiner_prompt")
print("REFINER_SYSTEM_PROMPT",REFINER_SYSTEM_PROMPT)

QA_SYSTEM_PROMPT = prompt = langfuse.get_prompt("qa_prompt")

print("QA_SYSTEM_PROMPT",QA_SYSTEM_PROMPT)



def call_stream(messages, context):
    user_question = messages[-1]['content'] if messages else ""
    
    prompt = QA_SYSTEM_PROMPT.compile(chunks=context, question=user_question)
    
    conversation_messages = [
        {"role": "system", "content": prompt}
    ]
    
    conversation_messages.extend(messages)
    
    try:
        stream = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=conversation_messages,
            stream=True,
            temperature=0.1,
            max_tokens=2000
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logging.error(f"OpenAI API call failed: {e}")
        yield f"Error: Unable to process your request. Please try again."


def call_refiner_prompt(messages, question):
    prompt = REFINER_SYSTEM_PROMPT.compile(conversation=messages, question=question)
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logging.error(f"Query refinement failed: {e}")
        return question

def cohere_rerank(query, docs, top_n=6):
    cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    results = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=top_n,
    )

    indexes = [i.index for i in results.results]
    texts = [docs[i] for i in indexes]
    relevance_scores = [i.relevance_score for i in results.results]

    return texts

def message_to_str(messages):
    messages_str = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        messages_str += f"{role}: {content}\n"
    return messages_str