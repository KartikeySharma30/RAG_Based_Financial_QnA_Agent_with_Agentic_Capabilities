#!/usr/bin/env python3
"""
Financial RAG Assistant Demo
Quick demonstration of the system's capabilities
"""

import asyncio
import os
from dotenv import load_dotenv
from src import FinancialRAGAssistant

# Load environment variables
load_dotenv()

async def demo_quick_start():
    """Quick demo with sample queries"""
    
    print("🚀 Financial RAG Assistant Demo")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in .env file")
        return
    
    try:
        # Initialize assistant
        print("\n📋 Initializing Financial RAG Assistant...")
        assistant = FinancialRAGAssistant()
        
        # Check if we have processed data already
        print("\n🔍 Checking for existing processed data...")
        if await assistant.load_from_processed_data():
            print("✅ Loaded existing processed data")
        else:
            print("📊 No existing data found. Starting fresh extraction...")
            
            # For demo, let's process just 2023 data to be faster
            companies = ['GOOGL', 'MSFT', 'NVDA']
            years = [2023]  # Limited to 2023 for demo speed
            
            print(f"\n🔄 Processing data for {companies} in {years}...")
            print("⏱️ This may take a few minutes for the first run...")
            
            success = await assistant.extract_and_process_filings(companies, years)
            if not success:
                print("❌ Failed to extract filings. Check network connection and try again.")
                return
            
            success = await assistant.setup_vector_database()
            if not success:
                print("❌ Failed to setup vector database. Check Qdrant connection.")
                return
        
        # Show system status
        print("\n📊 System Status:")
        status = assistant.get_system_status()
        print(f"  ✅ Initialized: {status['initialized']}")
        print(f"  📄 Documents: {status['processed_documents']}")
        print(f"  📝 Chunks: {status['document_chunks']}")
        print(f"  🏢 Companies: {status['available_companies']}")
        print(f"  📅 Years: {status['available_years']}")
        
        # Demo queries
        print("\n🤖 Running Demo Queries...")
        print("-" * 30)
        
        demo_queries = [
            "What was Microsoft's total revenue in 2023?",
            "How did NVIDIA's data center revenue perform in 2023?",
            "Which company had better profit margins in 2023 - Google or Microsoft?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n{i}. ❓ Query: {query}")
            print("   🔄 Processing...")
            
            try:
                answer = await assistant.answer_query(query)
                print(f"   🤖 Answer: {answer}")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
            
            print("   " + "-" * 40)
        
        # Show available query examples
        print("\n📚 More Query Examples You Can Try:")
        examples = assistant.get_query_examples()
        for category, queries in examples.items():
            print(f"\n{category}:")
            for query in queries:
                print(f"  • {query}")
        
        print("\n✅ Demo completed successfully!")
        print("\n💡 To run interactive mode, use: python main.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please check your configuration and try again.")

async def demo_interactive():
    """Interactive demo mode"""
    
    print("🤖 Interactive Financial RAG Assistant")
    print("=" * 40)
    
    try:
        assistant = FinancialRAGAssistant()
        
        # Try to load existing data first
        if not await assistant.load_from_processed_data():
            print("\nNo processed data found. Please run the full setup first:")
            print("python demo.py --quick-start")
            return
        
        print("\n✅ System ready! Ask your financial questions.")
        print("💡 Type 'examples' to see sample queries")
        print("💡 Type 'status' to see system information")
        print("💡 Type 'quit' to exit")
        
        while True:
            query = input("\n❓ Your question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'examples':
                examples = assistant.get_query_examples()
                for category, queries in examples.items():
                    print(f"\n{category}:")
                    for q in queries:
                        print(f"  • {q}")
                continue
            
            if query.lower() == 'status':
                status = assistant.get_system_status()
                print(f"\n📊 System Status:")
                for key, value in status.items():
                    print(f"  {key}: {value}")
                continue
            
            if not query:
                continue
            
            try:
                print("🔄 Processing...")
                answer = await assistant.answer_query(query)
                print(f"\n🤖 Answer:\n{answer}")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
    
    except Exception as e:
        print(f"❌ Interactive mode failed: {str(e)}")

def main():
    """Main demo function"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(demo_interactive())
    else:
        asyncio.run(demo_quick_start())

if __name__ == "__main__":
    main()