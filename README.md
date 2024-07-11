# Adaptive-Recommendation-Chatbot-with-RAG-and-Vector-Database# AI Information Chatbot

## Overview
This Streamlit-based chatbot application utilizes AI to provide personalized information from AI-related documents using vector embeddings and a question-answering system. It leverages PyMuPDF for PDF text extraction, Pinecone for vector storage and retrieval, and OpenAI's GPT-4 model for natural language processing.

## Features
- **PDF Text Extraction:** Extracts text from PDF documents using PyMuPDF.
- **Vector Embeddings:** Uses OpenAI's embeddings model for document representation.
- **Question-Answering:** Implements a retrieval and question-answering chain to respond to user queries based on the context stored in the document vectors.
- **Chat History Management:** Manages user interactions and maintains chat history across sessions.

## Installation
1. Clone the repository:
git clone [[https://github.com/hetalgada15/Adaptive-Recommendation-Chatbot-with-RAG-and-Vector-Database.git]
cd Adaptive-Recommendation-Chatbot-with-RAG-and-Vector-Database


2. Install dependencies from `requirements.txt`:

## Setup
1. **Environment Variables:**
- Ensure the following environment variables are set:
  - `OPENAI_API_KEY`: API key for OpenAI's GPT-4 model.
  - `PINECONE_API_KEY`: API key for Pinecone vector storage.
  - Optionally, set `PINECONE_ENV` to specify the Pinecone region (default: `us-east-1`).

2. **Run the Application:**
streamlit run AI.py


## Usage
- Access the chatbot interface through your web browser (default URL: `http://localhost:8501`).
- Enter your questions in the input box and click 'Send' to receive responses from the AI chatbot.

## Additional Notes
- **Export APIs:** The project includes functionalities to export APIs, providing an interface for integrating with other applications.
- **Chat History:** The application retains chat history for each session, allowing users to view previous interactions.

## Technologies Used
- Python
- Streamlit
- PyMuPDF
- Pinecone
- OpenAI GPT-4

## License
This project is licensed under the [MIT License](LICENSE).
