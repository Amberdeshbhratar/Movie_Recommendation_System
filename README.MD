# Movie Recommendation System

A conversational AI-powered movie recommender that helps users discover movies based on their interests, leveraging LLMs and embeddings for intelligent, context-aware suggestions.

## Features

- **Conversational Movie Recommendations:** Get movie suggestions through a chat interface.
- **Contextual Understanding:** Uses chat history to refine and personalize recommendations.
- **Top 1000 IMDb Movies:** Recommendations are based on a curated dataset of top-rated movies.
- **Modern Stack:** Built with FastAPI (backend) and React (frontend) for a seamless user experience.
- **LLM Integration:** Utilizes Google Gemini LLM via LangChain for natural language understanding and response generation.
- **Embeddings & Vector Search:** Employs HuggingFace embeddings and FAISS for semantic search over movie data.
- **Structured Responses:** Returns a brief AI message and a list of recommended movies (title, description, rating, poster).

## Demo

<!-- Replace with actual screenshot path -->

## Project Structure

Movie_Recommendation_System/ <br/>
├── Backend/ <br/>
│ ├── main.py <br/>
│ └── imdb_top_1000.csv <br/>
├── Frontend/ <br/>
│ ├── Chatui.jsx <br/>
│ └── index.css <br/>
├── LLM_model.ipynb <br/>
└── README.md <br/>

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js & npm
- [Google Gemini API Key](https://ai.google.dev/)

### Backend Setup

1. **Install dependencies:**
```
cd Backend
pip install fastapi uvicorn pandas langchain langchain-google-genai faiss-cpu huggingface-hub
```

2. **Set up API key:**
- Add your Google Gemini API key in `main.py` (`google_api_key`).

3. **Run the FastAPI server:**
```
uvicorn main:app --reload
```

### Frontend Setup

1. **Install dependencies:**
```
cd Frontend
npm install
```

2. **Start the React app:**
```
npm run dev
```

3. **Access the UI:**  
Navigate to `http://localhost:5173` in your browser.

## API

### POST `/chat`

**Request:**
```
{
"message": "Suggest me a sci-fi movie",
"chat_history": ["Suggest me a comedy", "Now a sci-fi"]
}
```
**Response:**
```
{
"answer": "Here are some top-rated sci-fi movies you might like.",
"movie_list": [
{
"title": "Inception",
"description": "A thief who steals corporate secrets through dream-sharing technology...",
"rating": 8.8,
"url": "https://..."
}
]
}
// up to 5 movies
```
## Tech Stack

| Layer      | Technology                         |
|------------|------------------------------------|
| Backend    | FastAPI, LangChain, FAISS, Pandas  |
| LLM        | Google Gemini (via LangChain)      |
| Embeddings | HuggingFace Transformers           |
| Frontend   | React, Axios, Lucide-React Icons   |
| Data       | IMDb Top 1000 Movies (CSV)         |

## How It Works

- Loads the IMDb dataset and converts each movie to a LangChain document.
- Embeds documents using HuggingFace models, stores them in FAISS for fast vector search.
- Handles chat via FastAPI, maintaining conversation history for context.
- Uses Gemini LLM to generate responses and recommend movies in structured JSON.
- The React frontend communicates with the backend, displaying chat and movie cards.

## Customization

- Replace `imdb_top_1000.csv` with your own dataset for different recommendations.
- Update the LLM or embedding model in `main.py` for improved performance.

## Acknowledgments

- [LangChain](https://python.langchain.com/)
- [HuggingFace](https://huggingface.co/)
- [IMDb Datasets](https://www.imdb.com/interfaces/)
- [Google Gemini](https://ai.google.dev/)

---

> **GitHub:** [Amberdeshbhratar/Movie_Recommendation_System](https://github.com/Amberdeshbhratar/Movie_Recommendation_System)
