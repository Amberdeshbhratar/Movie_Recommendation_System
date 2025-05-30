import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import json

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain_core.prompts import ChatPromptTemplate


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


format_instruction = """Respond with a JSON object containing:
- "ai_message": brief text response
- "movie_list": array of movie objects with "title", "description", "rating", "url\""""

template = f"""
You are a movie recommender expert. Use this context:
{{context}}

User question: {{question}}
Chat history: {{chat_history}}

{format_instruction}

Return ONLY valid JSON, no extra text or markdown.
"""

# Create question rephrasing prompt
condense_question_prompt = ChatPromptTemplate.from_template("""
Given this chat history:
{chat_history}

And a follow-up question: {question}
Rephrase the follow-up question to be standalone, incorporating any necessary context from the chat history.

Standalone question:""")

# Prompt Template
prompt = PromptTemplate(
    input_variables=["question", "chat_history" , "context"],  # Remove format_instruction
    template=template
)

# Conversation memory
# Memory Setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    k=100,
    output_key="raw_output"
)

# LLM setup
llm = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyCpg4ohn2ORebEQY_p7Fvwz1PB4yppxE5k",
    model="gemini-2.0-flash"
)

# QA Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True,
    output_key="raw_output",
    condense_question_prompt=condense_question_prompt,  # Context prompt
    # verbose=True,  # <-- Debugging
    combine_docs_chain_kwargs={"prompt": prompt}  # <--  Promt
)





# Movie info model
class MovieInfo(BaseModel):
    title: str = Field(..., description="This is the Series_Title of the movie.")
    description: str = Field(..., description="This is a short description of the movie.")
    rating: float = Field(..., description="Decimal rating between 1-10.")
    url: str = Field(..., description="URL of the poster image.")

# Request and response models
class ChatRequest(BaseModel):
    message: str
    chat_history: List[str]

class ChatResponse(BaseModel):
    answer: str
    movie_list: List[MovieInfo]



# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("every thing done")
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    print(req)
    try:
        result = qa_chain.invoke({"question": req.message})
        raw_output = result.get("raw_output", "")

        # Trim the response (you know your LLM returns JSON like this)
        trimmed = raw_output[8:-4]

        # Parse it to dict
        parsed = json.loads(trimmed)

        # Extract fields
        ai_message = parsed.get("ai_message", "Sorry, no explanation provided.")
        movie_list_raw = parsed.get("movie_list", [])

        # Convert each dict to MovieInfo object
        movie_list = []
        for movie_data in movie_list_raw:
            try:
                movie = MovieInfo(**movie_data)
                print(movie)
                movie_list.append(movie)
            except Exception as e:
                print(f"Skipping a movie due to error: {e}")
                continue

        return ChatResponse(answer=ai_message, movie_list=movie_list[:5])

    except Exception as e:
        print(f"Error occurred: {e}")
        return ChatResponse(answer="Error occured", movie_list=[])
