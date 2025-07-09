import streamlit as st
from openai import OpenAI
import pandas as pd
import faiss
import duckdb
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel
from typing import Literal, Optional
from openai import OpenAI
import re

api_key= st.secrets["OPEN_AI_KEY"]
# Initialize OpenAI Client
client = OpenAI(api_key= api_key)

print("✅ FAISS loaded successfully!")

# Load Vector Store (for SOP/Policies)
VECTOR_DB_PATH = "./sop_faiss_index/"
embedding = OpenAIEmbeddings(api_key=api_key)

vector_store = None  # Ensure it's always defined
try:
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)
    print("✅ FAISS loaded successfully!")
except Exception as e:
    print("⚠️ Error loading FAISS:", e)

print(f"Vector store initialized? {vector_store is not None}")

# Define CSV Inventory Data Directory
CSV_DIR = "inventory_data"

#List of CSV files
csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]

# Function to retrieve policy documents
def retrieve_policy(query: str):
    if vector_store is None:
        return "⚠️ Policy retrieval system is currently unavailable. Please try again later."

    docs = vector_store.similarity_search(query, k=1)  # Retrieve top 3 relevant docs
    if docs:
        return "\n\n".join([doc.page_content for doc in docs])

    return "❌ No relevant policy found."

def extract_date_llm(query: str) -> str:
    """Uses LLM to extract and format date in YYYYMM."""

    prompt = f"""
    Extract and return only the date from the following query in YYYYMM format. 
    If the date is relative (e.g., "last month"), convert it to absolute form based on today’s date.

    Query: "{query}"
    
    match the date to one of the csv file names {csv_files}

    Only return the original file name nothing else. Just one word.
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": prompt}])

    extracted_date = response.choices[0].message.content.strip()
    extracted_date = extracted_date.strip("'")
    
    return extracted_date

def extract_category_from_query(query: str) -> Optional[str]:
    """Extract category keyword from the query."""
    category_keywords = ["electronics", "clothing", "furniture", "books", "toys"]  # Expand as needed
    words = query.lower().split()

    for word in words:
        if word in category_keywords:
            return word
    return None

def process_inventory_query(query: str):
    """Processes an inventory query using DuckDB and Pandas."""
    if not os.path.exists(CSV_DIR):
        return "⚠️ Inventory data directory not found."

    # Extract date and category from query
    date = extract_date_llm(query)
    category = extract_category_from_query(query)
    #List of CSV files
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
    # Filter files by date if provided
    if date:
        csv_files = [f for f in csv_files]
        print(csv_files)
        csv_files = [f for f in csv_files if date in f]
        print(csv_files)

    if not csv_files:
        return "❌ No inventory data found for the specified date."

    # Sort and load latest CSV
    latest_csv = os.path.join(CSV_DIR, sorted(csv_files)[-1])

    # Use DuckDB for efficient querying
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE TABLE inventory AS SELECT * FROM read_csv_auto('{latest_csv}')")
    
    query_str = f"SELECT '{category.lower()}', SUM(quantity) AS total_quantity FROM inventory"
    print(query_str)
    if category:
        query_str += f" WHERE LOWER(category) LIKE '%{category.lower()}%'"

    query_str += "GROUP BY category ORDER BY total_quantity DESC"

    # Run the query
    result = con.execute(query_str).fetchdf()

    if result.empty:
        return "❌ No matching inventory records found."

    return result
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
        st.write("🔍 Analyzing inventory query...")
        extracted_date = extract_date_llm(query_text)
        extracted_category = extract_category_from_query(query_text)
        result = process_inventory_query(query_text)

    # Process policy queries
    elif "policy" in query_type:
        st.write("📜 Retrieving policy details...")
        result = retrieve_policy(str(query_text))

    else:
        result = "⚠️ I couldn't understand the query type."

    st.write(result)
