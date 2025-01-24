from typing import Annotated
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
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