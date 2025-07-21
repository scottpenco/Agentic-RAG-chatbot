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

# Load Vector Store (for SOP/Policies)
VECTOR_DB_PATH = "sop_faiss_index"
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

    docs = vector_store.similarity_search(query, k=1)  # Retrieve top 1 relevant docs
    if docs:
        return "\n\n".join([doc.page_content for doc in docs])

    return "❌ No relevant policy found."

def extract_date_llm(query: str) -> str:
    """Uses LLM to extract and format date in YYYYMM."""
    prompt = f"""
    You are an expert data assistant helping to match natural language queries to a set of monthly CSV data files.

    Your task is to:
    1. Extract a **single date** from the query.
    2. If the date is relative (e.g., "last month", "this year"), convert it to **absolute YYYYMM** format.
    3. Match this date to one of the available CSV file names: {csv_files}.
    4. Return **only** the exact matching file name. No explanations, no extra text.

    ## Input
    Query: "{query}"

    ## Output
    One word only: the matching CSV filename (e.g., 'inventory_feburary_2024.csv') or nothing if no match.
    """

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": prompt}])

    extracted_date = response.choices[0].message.content.strip()
    extracted_date = extracted_date.strip("'")
    
    return extracted_date

def extract_sql_query(query: str) -> str:
    """Uses LLM to extract or convert the user requests into a SQL query for an 'invetory' table"""
    prompt = f"""
    You are a AI assistant helping to match user requests into a SQL query for an 'invetory' table.

    The 'inventory' table has the following columns:
    - date (YYYY-MM-DD)
    - category (TEXT) - e.g., 'Electronics', 'Apparel', 'Home Goods', 'Footwear'
    - product (TEXT) -  e.g., 'Smartphones', 'Jackets', 'Rugs', 'Boots'
    - quantity (INT)
    - location (TEXT) - 'Warehouse A', 'Warehouse B', 'Warehouse C'

    Only use fields from the table. Do not invent any columns. Use safe lowercase LIKE statements for text matching when needed.

    Examples: 
    User: "How many boots are available?"  
    SQL: SELECT SUM(quantity) FROM inventory WHERE LOWER(product) LIKE '%boots%';

    User: "What’s the stock level for rugs in Warehouse C?"  
    SQL: SELECT SUM(quantity) FROM inventory WHERE LOWER(product) LIKE '%rugs%' AND LOWER(location) = 'warehouse c';

    User: "Do you have any jackets in stock?"  
    SQL: SELECT * FROM inventory WHERE LOWER(product) LIKE '%jackets%' AND quantity > 0;

    User: "What products are available in Warehouse A?"  
    SQL: SELECT DISTINCT product FROM inventory WHERE LOWER(location) = 'warehouse a' AND quantity > 0;

    Don't create about the date, another agent takes care of this 

    User: "Show inventory for February 2024"  
    SQL: SELECT * FROM inventory;
    Now write an SQL query for the following:

    User: "{query}"

    Only output the SQL query."""

    response = client.chat.completions.create(model="gpt-4",
    messages=[{"role": "system", "content": prompt}])
    sql_query = response.choices[0].message.content

    return sql_query

def process_inventory_query(query: str):
    """Processes an inventory query using DuckDB and Pandas."""
    if not os.path.exists(CSV_DIR):
        return "⚠️ Inventory data directory not found."

    # Extract date and category from query
    date = extract_date_llm(query)

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
    query_str = extract_sql_query(query)
    print(query_str)
    result = con.execute(query_str).fetchdf()

    if result.empty:
        return "❌ No matching inventory records found."

    return result
# Streamlit UI
st.title("🤖💬I'm a multi-agent chatbot designed to answer Inventory or Policy questions")

# User Query Input
query_text = st.text_input("Ask me about inventory shipments or policies:")

if st.button("Submit") and query_text:
    # Determine if query is for inventory or policy
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that classifies user queries"},
            {"role": "user", "content": f"""
                You are a router agent in a multi-agent system. Your job is to classify user queries as either related to `inventory` or `policy`. 

                Inventory queries are about product stock, availability, sizes, or restocks.  
                Policy queries refer to company rules like returns, exchanges, pricing policies, or store hours.

                Examples:
                - "Do you have size 8 in stock?" → inventory  
                - "What's your return window?" → policy  
                - "Is the blue jacket available?" → inventory  
                - "Do you price match other stores?" → policy  

                Classify the following query with just one word: `inventory` or `policy`. Do not explain your reasoning.

                Query: "{query_text}"
        """},
        ],
    )

    query_type = response.choices[0].message.content.strip().lower()

    # Process inventory queries
    if "inventory" in query_type:
        st.write("🔍 Analyzing inventory query...")
        result = process_inventory_query(query_text)

    # Process policy queries
    elif "policy" in query_type:
        st.write("📜 Retrieving policy details...")
        result = retrieve_policy(str(query_text))

    else:
        result = "⚠️ I couldn't understand the query type."

    st.write(result)
