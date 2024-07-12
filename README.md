# PDF Information Chatbot

![Chatbot Demo](demo.gif)

## Overview

This repository contains an AI-powered PDF Information Chatbot developed using Python, Streamlit, PyMuPDF, OpenAI's GPT-3.5-turbo model, Pinecone for vector indexing, and other libraries. The chatbot allows users to upload PDF documents, extracts and indexes their content, and provides intelligent responses to user queries based on document content similarity.

## Features

- **PDF Upload**: Users can upload multiple PDF documents.
- **Document Processing**: Extracts text content from uploaded PDFs using PyMuPDF.
- **Vector Embeddings**: Converts text content into high-dimensional vector representations using OpenAI's embeddings.
- **Pinecone Indexing**: Indexes embeddings for fast retrieval and recommendation.
- **Interactive Chat Interface**: Allows users to query information from uploaded PDFs via a Streamlit-based chat interface.
- **Recommendation System**: Recommends relevant PDFs based on user queries and document content similarity.
- **Learning and Adaptation**: System learns from user interactions to improve recommendations over time.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/pdf-information-chatbot.git
   cd pdf-information-chatbot
   
2. **Install dependencies:**

pip install -r requirements.txt

3. **Set up environment variables:**

OPENAI_API_KEY: Your OpenAI API key for GPT-3.5-turbo model.
PINECONE_API_KEY: Your Pinecone API key.
PINECONE_ENV (optional): Pinecone environment (default is us-east-1)

4. **Run the application:**
streamlit run AI.py

5. **Open your browser and go to http://localhost:8501 to access the chatbot interface.**

Usage
Upload PDF files using the file uploader.
Select a PDF file to query or search.
Enter your question in the text input box and click "Send" to get a response from the chatbot.
Use the "Clear" button to reset the chat history and messages.
Contributing
Contributions are welcome! Please feel free to open issues or pull requests for any improvements, features, or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
PyMuPDF - For PDF document processing.
OpenAI - For providing the GPT-3.5-turbo model.
Pinecone - For vector indexing and fast retrieval.
Streamlit - For building interactive web applications with Python.


## This Markdown document provides clear sections for overview, features, installation instructions, usage guidelines, contributing guidelines, license information, and acknowledgments. Adjust the URLs, paths, and descriptions as per your specific project details.
