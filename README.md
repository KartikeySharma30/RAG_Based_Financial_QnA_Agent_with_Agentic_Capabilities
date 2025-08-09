#  **RAG-Based Financial Q&A Agent with Agentic Capabilities**

## Overview

- This project implements a Retrieval-Augmented Generation (RAG)–based Financial Q&A Agent capable of answering complex queries from SEC 10-K filings of Google, Microsoft, and NVIDIA (2022–2024). The system combines semantic search with LLM-based reasoning to deliver accurate, context-aware answers from large financial documents.

- With agentic capabilities, the system performs query decomposition, multi-step retrieval, and synthesis to handle nuanced financial questions involving both text and tabular data.

## How to Run/Execute :
> Update_file -> RAG... .ipynb

## Objective 

- Enable intelligent financial question answering using SEC 10-K filings.
- Implement vector search for efficient retrieval of relevant text and table data.
- Handle complex queries via multi-step reasoning and synthesis.
- Integrate structured data from tables into LLM responses for higher accuracy.

## Data Source
- **SEC EDGAR** : <u>https://www.sec.gov/search-filings</u>
- **Documents** : SEC 10-K filings (2022–2024) for Google, Microsoft, NVIDIA
- **Format** : Text (in which Extracted and used (Consisted of Textual and Tablular Data)) , Original Doc : PDF (contains Text , Tables , Images )
- **Contents** : Text, tables, and financial statements

## Tech Stack
- Languages: Python
- Libraries: LangChain, Hugging Face Sentence Transformers, Pandas, edgartools
- Vector Database: Qdrant (selected over FAISS/Chroma for hybrid search & persistence)
- LLM: Google Gemini

## Workflow
- **Data Ingestion**
  > Download SEC filings from EDGAR using edgartools
  
  > Parse text 
  
  > Clean and structure extracted data

- **Chunking & Embedding**
  > Chunk text using CharacterTextSplitter

  > Generate embeddings with sentence-transformers/all-MiniLM-L6-v2

  > Store in Qdrant vector database

- **Retrieval Pipeline**
  > Perform semantic search to retrieve relevant chunks

  > Retrieve both narrative text and associated tables as metadata

- **Agentic Query Processing**

  > Decompose complex queries into sub-queries**

  > Perform multi-step retrieval & synthesis

  > Merge textual and tabular insights into a single response

- **Answer Generation**

  > Pass retrieved context to the LLM

  > Generate a concise, accurate answer with references to source sections

## **Features**
  > Hybrid Retrieval: Supports both text and table lookups.

  > Multi-step Reasoning: Breaks down complex queries for better accuracy.

  > Agentic Capabilities: Dynamically decides retrieval steps.

  > Context Awareness: Includes related tables and metadata in responses.

