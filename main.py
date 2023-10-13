"""Main entrypoint for the app."""
import os
from typing import List, Union, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langserve import add_routes
from typing_extensions import TypedDict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(TypedDict):
    question: str
    chat_history: Union[List[Tuple[str, str]], None]


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


prompt = ChatPromptTemplate.from_template(
    """You are a pirate named Patchy. Respond to all questions in pirate dialect.

Current conversation:
{chat_history}
Human: {question}
AI:"""
)

runnable = (
    RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"] or [])
    )
    | prompt
    | ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
    | StrOutputParser()
)

add_routes(app, runnable, path="/chat", input_type=ChatRequest)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
