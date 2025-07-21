# 📦 Inventory & Policy RAG Chatbot

An AI-powered chatbot built with Streamlit and OpenAI that allows users to query company **inventory shipments** and retrieve relevant **policy documents** using **Retrieval-Augmented Generation (RAG)**.

---

## Features

-  **Query Classification** – Detects if a query is about inventory or policy using GPT-4.
-  **Inventory Insights** – Loads CSV shipment data and analyzes it using DuckDB and Pandas.
-  **Policy Retrieval** – Finds relevant SOP/policy documents using vector search with FAISS.
-  **Natural Language Interface** – Built with Streamlit for a clean and simple UI.
-  **Semantic Search** – Uses OpenAI embeddings for policy search with context awareness.

## 🤖 Agent Architecture

This application is structured around a simple multi-agent setup, with each agent responsible for a distinct role:

| Agent | Role | Description |
|-------|------|-------------|
| 🔍 **Query Classifier Agent** | **Classification** | Uses GPT-4 to determine whether a user query is about _inventory_ or _policy_. This routing step ensures each query is handled by the correct agent. |
| 📊 **Inventory Data Agent** | **Structured Query Processing** | Extracts date/category from user queries and runs SQL-like queries using **DuckDB** and **Pandas** on CSV-based inventory data. |
| 📄 **Policy Retrieval Agent** | **Unstructured Search** | Uses **FAISS vector search** and **OpenAI Embeddings** to find the most relevant company policy or SOP for a given query. |

This modular approach allows for easy extension into a full multi-agent framework.

![alt text](https://ibb.co/5gZkNGzT](https://i.ibb.co/mVp0LbYJ/Multi-AIRAGPolicy.png)



