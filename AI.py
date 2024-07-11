


import os
import fitz  # PyMuPDF
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import cachetools
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Concrete implementation of BaseChatMessageHistory
class SimpleChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, message):
        self.messages.append(message)

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages.clear()

# Set up the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")

# Initialize Pinecone with API key
pc = Pinecone(api_key=pinecone_api_key)

# Create or connect to a Pinecone index
index_name = "ai-index"
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
index = pc.Index(index_name)

# Initialize OpenAI embeddings model with API key
embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        return ""

# Asynchronously load and chunk PDF documents
async def load_and_chunk_pdfs(pdf_paths, chunk_size):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        tasks = [
            loop.run_in_executor(executor, functools.partial(extract_text_from_pdf, pdf_path))
            for pdf_path in pdf_paths
        ]
        texts = await asyncio.gather(*tasks)
    
    documents = [Document(page_content=text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# Convert list of documents to a hashable type for caching
def convert_to_hashable(documents):
    return tuple((doc.page_content for doc in documents))

# Caching results to improve performance
@cachetools.cached(cache=cachetools.LRUCache(maxsize=128), key=lambda docs: convert_to_hashable(docs))
def get_cached_vector_store(split_docs):
    from langchain_pinecone import PineconeVectorStore
    return PineconeVectorStore.from_documents(split_docs, embeddings_model, index_name=index_name)

# Define system and QA prompts
system_prompt = (
    "You are an assistant specialized in providing personalized information from AI-related documents. "
    "Answer the user's questions based on the context provided from the vector database. "
    "If the context does not have relevant information, guide the user to ask relevant questions or suggest alternative approaches."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

model_name = "gpt-4"
llm = ChatOpenAI(model_name=model_name)

# Set up the retrieval chain
def get_retrieval_chain(vector_store):
    retriever = create_history_aware_retriever(
        llm=llm, retriever=vector_store.as_retriever(), prompt=contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )

    st.session_state.chat_active = True

    return rag_chain

store = {}

def get_session_history(session_id: str) -> SimpleChatMessageHistory:
    if session_id not in store:
        store[session_id] = SimpleChatMessageHistory()
    return store[session_id]


def get_answer(query):
    retrieval_chain = get_retrieval_chain(st.session_state.vector_store)
    st.session_state.history.append({"role": "user", "content": query})
    session_id = "session_id"
    history = get_session_history(session_id)
    answer = retrieval_chain.invoke({"input": query, "chat_history": history.get_messages()})
    
    if answer["answer"].startswith("The context provided does not contain specific information"):
        response = "This question is irrelevant to the document. Sorry!"
    else:
        response = answer["answer"]
    
    st.session_state.history.append({"role": "assistant", "content": response})
    
    return {"answer": response.split("\n")[0]}

# Streamlit app setup
st.title("🔗 AI Information Chatbot")

if "vector_store" not in st.session_state:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    split_docs = loop.run_until_complete(load_and_chunk_pdfs(["AI.pdf"], chunk_size=1000))
    st.session_state.vector_store = get_cached_vector_store(split_docs)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.expander(message["role"]):
        st.write(message["content"])

# React to user input
query = st.text_input("Ask your question here:")
if st.button("Send"):
    answer = get_answer(query)
    with st.expander("Assistant"):
        st.write(answer["answer"])

def clear_messages():
    st.session_state.messages = []
    st.session_state.history = []  # Clear the conversation history as well
st.button("Clear", help="Click to clear the chat", on_click=clear_messages)

# Footer
st.sidebar.text("Powered by OpenAI and Pinecone")

