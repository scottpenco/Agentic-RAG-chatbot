import streamlit as st
import openai
import pandas as pd
import faiss
import duckdb
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pydantic import BaseModel
from typing import Literal, Optional
from openai import OpenAI

# Initialize OpenAI Client
client = OpenAI(api_key=st.secrets["OPEN_AI_KEY"])

# Set OpenAI API Key
openai.api_key = st.secrets["OPEN_AI_KEY"]

# Load Vector Store (for SOP/Policies)
VECTOR_DB_PATH = "./sop_faiss_index/"
embedding = OpenAIEmbeddings(api_key=st.secrets["OPEN_AI_KEY"])


try:
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)
    print("✅ FAISS loaded successfully!")
except Exception as e:
    print("⚠️ Error loading FAISS:", e)

print(f"Vector store initialized? {vector_store is not None}")

# Define CSV Inventory Data Directory
CSV_DIR = "inventory_data"

# Define Query Schema
class QueryRequest(BaseModel):
    query_type: Literal["inventory", "policy"]
    query_text: str
    date: Optional[str] = None  # For filtering inventory
    category: Optional[str] = None  # For filtering inventory

# Function to retrieve policy documents
def retrieve_policy(query: str):
    if vector_store is None:
        return "⚠️ Policy retrieval system is currently unavailable. Please try again later."

    docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 relevant docs
    if docs:
        return "\n\n".join([doc.page_content for doc in docs])
    
    return "❌ No relevant policy found."

# Function to process inventory CSV queries
def process_inventory_query(date: Optional[str], category: Optional[str]):
    if not os.path.exists(CSV_DIR):
        return "⚠️ Inventory data directory not found."

    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]

    # Find latest CSV matching the requested date
    if date:
        csv_files = [f for f in csv_files if date in f]

    if not csv_files:
        return "❌ No inventory data found for the specified date."

    # Load the latest matching CSV
    latest_csv = os.path.join(CSV_DIR, csv_files[-1])
    df = pd.read_csv(latest_csv)

    # Apply category filtering if requested
    if category:
        df = df[df["category"].str.contains(category, case=False, na=False)]

    # Aggregate shipment summary
    summary = df.groupby("category")["quantity"].sum().reset_index()
    
    return summary.to_string(index=False)

# Streamlit UI
st.title("📦 Inventory & Policy RAG Chatbot")

# User Query Input
query_text = st.text_input("Ask me about inventory shipments or policies:")

if st.button("Submit") and query_text:
    # Determine if query is for inventory or policy
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Classify the query as either 'inventory' or 'policy'."},
            {"role": "user", "content": query_text},
        ],
    )

    query_type = response.choices[0].message.content.strip().lower()

    # Process inventory queries
    if "inventory" in query_type:
        st.write("🔍 Searching inventory records...")
        result = process_inventory_query(date="2024-03", category=None)

    # Process policy queries
    elif "policy" in query_type:
        st.write("📜 Retrieving policy details...")
        st.write("You entered:", query_text)
        result = retrieve_policy(str(query_text))
        try:
            vector_store = FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)
            print("✅ FAISS loaded successfully!")
        except Exception as e:
            print("⚠️ Error loading FAISS:", e)

            print(f"Vector store initialized? {vector_store is not None}")

    
    else:
        result = "⚠️ I couldn't understand the query type."

    st.write(result)
