#!/usr/bin/env python3
"""
Financial RAG Assistant - Main Entry Point
Extracts SEC filings, processes financial data, and provides intelligent Q&A
"""

import os
import asyncio
from dotenv import load_dotenv
from src.rag_assistant import FinancialRAGAssistant

# Load environment variables
load_dotenv()

async def main():
    """Main function to run the Financial RAG Assistant"""
    
    # Initialize the RAG assistant
    assistant = FinancialRAGAssistant()
    
    # Companies and years to process
    companies = ['GOOGL', 'MSFT', 'NVDA']  # Google, Microsoft, NVIDIA
    years = [2022, 2023, 2024]
    
    print("🚀 Starting Financial RAG Assistant...")
    
    # Step 1: Extract and process SEC filings
    print("\n📊 Extracting SEC 10-K filings...")
    await assistant.extract_and_process_filings(companies, years)
    
    # Step 2: Initialize vector database
    print("\n🔍 Setting up vector database...")
    await assistant.setup_vector_database()
    
    # Step 3: Start interactive Q&A session
    print("\n💬 Starting interactive Q&A session...")
    print("Type 'quit' to exit.")
    
    while True:
        query = input("\n❓ Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        try:
            response = await assistant.answer_query(query)
            print(f"\n🤖 Answer: {response}")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())