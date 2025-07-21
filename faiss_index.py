import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


# Load Policy/SOP Documents
DATA_PATH = os.path.abspath("./retail_sop_documents")
VECTOR_DB_PATH = "sop_faiss_index"

# Load and split documents
loader = DirectoryLoader(DATA_PATH, glob="*.txt")  # Assumes text files
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

# Convert to vector embeddings
embeddings = OpenAIEmbeddings(api_key= st.secrets("OPEN_AI_KEY")
)
vector_store = FAISS.from_documents(chunks, embeddings)

# Save FAISS index
vector_store.save_local(VECTOR_DB_PATH)
print("✅ Policy documents indexed in FAISS!")
