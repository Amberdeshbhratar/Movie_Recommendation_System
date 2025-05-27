import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema,StructuredOutputParser

# Load dataset
df = pd.read_csv('imdb_top_1000.csv')

# Convert rows to LangChain documents
documents = [
    Document(
        page_content="\n".join(f"{k}: {v}" for k, v in row.items())
    )
    for _, row in df.iterrows()
]

# Vector store with embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_function)

aimessage=ResponseSchema(name="title",description="this is the Message send by chat bot")
title=ResponseSchema(name="title",description="this is the Series_Title of the Movie")
description=ResponseSchema(name="description",description="this is the short description of the Movie")
rating=ResponseSchema(name="rating",description="this is whole integer,this gives the rating between 1-10")
url=ResponseSchema(name="url",description="this gives url of the image in poster")

response_schema=[aimessage,title,description,rating,url]

output_parser=StructuredOutputParser.from_response_schemas(response_schema)
format_instruction=output_parser.get_format_instructions()
ts="""
You are a movie recommender system that help users to find movie that match their preferences.
Use the following pieces of context to answer the question at the end.
For each question, suggest atmost three movie, with a short description of the plot and the reason why the user might like it.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Take the input below delimited by tripe backticks
input:```{input}```
{format_instruction}
"""

# Conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=100,
    output_key="answer"
)

# LLM setup
llm = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyCpg4ohn2ORebEQY_p7Fvwz1PB4yppxE5k",
    model="gemini-2.0-flash"
)

# QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
    output_key="answer"
)

# Request and response models
class ChatRequest(BaseModel):
    message: str
    chat_history: List[str]

class ChatResponse(BaseModel):
    answer: str
    

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat route using LangChain pipeline
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        response = qa_chain({"question": req.message})
        answer = response["answer"]
    except Exception as e:
        answer = "Sorry, an error occurred."
    return ChatResponse(answer=answer)