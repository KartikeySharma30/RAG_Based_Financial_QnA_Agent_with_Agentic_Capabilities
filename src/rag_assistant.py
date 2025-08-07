"""
Financial RAG Assistant
Main interface that integrates all components for comprehensive financial analysis
"""

import os
import asyncio
from typing import List, Dict, Optional
from .sec_extractor import SECExtractor
from .document_processor import DocumentProcessor
from .document_chunker import DocumentChunker
from .vector_store import VectorStore
from .multi_step_retriever import MultiStepRetriever

class FinancialRAGAssistant:
    """Main RAG Assistant for financial document analysis"""
    
    def __init__(self, 
                 data_dir: str = "/workspace/data",
                 processed_dir: str = "/workspace/processed_data",
                 openai_api_key: Optional[str] = None,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333):
        
        # Configuration
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        
        # API keys
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        # Initialize components
        self.sec_extractor = SECExtractor()
        self.document_processor = DocumentProcessor()
        self.document_chunker = DocumentChunker()
        
        # Vector store
        self.vector_store = VectorStore(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_api_key=os.getenv("QDRANT_API_KEY")
        )
        
        # Multi-step retriever
        self.retriever = MultiStepRetriever(
            vector_store=self.vector_store,
            openai_api_key=self.openai_api_key
        )
        
        # State tracking
        self.is_initialized = False
        self.processed_documents = []
        self.document_chunks = []
    
    async def extract_and_process_filings(self, companies: List[str], years: List[int]) -> bool:
        """Extract and process SEC filings for specified companies and years"""
        try:
            print("📊 Starting SEC filing extraction and processing...")
            
            # Step 1: Extract SEC filings
            print("\n1️⃣ Extracting SEC 10-K filings...")
            sec_save_dir = os.path.join(self.data_dir, "sec_filings")
            downloaded_files = self.sec_extractor.extract_all_filings(companies, years, sec_save_dir)
            
            if not downloaded_files:
                print("❌ No files were downloaded. Please check company tickers and years.")
                return False
            
            print(f"✅ Downloaded {len(downloaded_files)} filings")
            
            # Step 2: Process documents (text, tables, images)
            print("\n2️⃣ Processing documents...")
            self.processed_documents = []
            
            for filepath in downloaded_files:
                if os.path.exists(filepath):
                    print(f"Processing: {os.path.basename(filepath)}")
                    processed_doc = self.document_processor.process_document(filepath)
                    self.processed_documents.append(processed_doc)
                    
                    # Save processed document
                    processed_filename = os.path.basename(filepath).replace('.html', '_processed.json')
                    processed_filepath = os.path.join(self.processed_dir, processed_filename)
                    
                    os.makedirs(self.processed_dir, exist_ok=True)
                    import json
                    with open(processed_filepath, 'w', encoding='utf-8') as f:
                        # Save without the soup object (not JSON serializable)
                        save_doc = {k: v for k, v in processed_doc.items() if k != 'soup'}
                        json.dump(save_doc, f, indent=2, default=str)
            
            print(f"✅ Processed {len(self.processed_documents)} documents")
            
            # Step 3: Chunk documents
            print("\n3️⃣ Chunking documents...")
            self.document_chunks = self.document_chunker.chunk_multiple_documents(self.processed_documents)
            
            # Print chunking statistics
            stats = self.document_chunker.get_chunk_stats(self.document_chunks)
            print("📈 Chunking Statistics:")
            print(f"  Total chunks: {stats.get('total_chunks', 0)}")
            print(f"  Average chunk size: {stats.get('avg_chunk_size', 0):.0f} characters")
            print(f"  Company distribution: {stats.get('company_distribution', {})}")
            print(f"  Year distribution: {stats.get('year_distribution', {})}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error in extraction and processing: {str(e)}")
            return False
    
    async def setup_vector_database(self) -> bool:
        """Setup and populate the vector database"""
        try:
            print("\n🔍 Setting up vector database...")
            
            # Create collection
            created = self.vector_store.create_collection()
            if created:
                print("✅ Created new Qdrant collection")
            else:
                print("ℹ️ Using existing Qdrant collection")
            
            # Store document chunks
            if self.document_chunks:
                print(f"💾 Storing {len(self.document_chunks)} chunks...")
                success = self.vector_store.store_chunks(self.document_chunks)
                
                if success:
                    print("✅ Successfully stored all chunks in vector database")
                    
                    # Get and display collection statistics
                    stats = self.vector_store.get_collection_stats()
                    print("📊 Vector Database Statistics:")
                    print(f"  Total vectors: {stats.get('total_points', 0)}")
                    print(f"  Vector dimension: {stats.get('vector_size', 0)}")
                    print(f"  Companies: {list(stats.get('company_distribution', {}).keys())}")
                    
                    self.is_initialized = True
                    return True
                else:
                    print("❌ Failed to store chunks in vector database")
                    return False
            else:
                print("⚠️ No chunks available to store. Please run extract_and_process_filings first.")
                return False
                
        except Exception as e:
            print(f"❌ Error setting up vector database: {str(e)}")
            return False
    
    async def answer_query(self, query: str) -> str:
        """Answer a financial query using the RAG system"""
        try:
            if not self.is_initialized:
                return "❌ System not initialized. Please run setup first."
            
            print(f"\n💭 Processing query: {query}")
            
            # Process query through multi-step retriever
            context_synthesis = self.retriever.process_query(query)
            
            # Generate final answer
            print("🤖 Generating final answer...")
            answer = self.retriever.generate_final_answer(context_synthesis)
            
            return answer
            
        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_system_status(self) -> Dict:
        """Get current system status and statistics"""
        return {
            "initialized": self.is_initialized,
            "processed_documents": len(self.processed_documents),
            "document_chunks": len(self.document_chunks),
            "vector_store_stats": self.vector_store.get_collection_stats() if self.is_initialized else {},
            "available_companies": list(set(chunk.company for chunk in self.document_chunks)),
            "available_years": list(set(chunk.year for chunk in self.document_chunks))
        }
    
    def get_query_examples(self) -> Dict[str, List[str]]:
        """Get example queries that can be processed"""
        from .query_decomposer import QueryDecomposer
        decomposer = QueryDecomposer()
        return decomposer.get_query_examples()
    
    async def reset_system(self) -> bool:
        """Reset the entire system (clear database and processed data)"""
        try:
            print("🔄 Resetting system...")
            
            # Delete vector collection
            if self.is_initialized:
                self.vector_store.delete_collection()
            
            # Clear processed data
            self.processed_documents = []
            self.document_chunks = []
            self.is_initialized = False
            
            print("✅ System reset complete")
            return True
            
        except Exception as e:
            print(f"❌ Error resetting system: {str(e)}")
            return False
    
    async def load_from_processed_data(self) -> bool:
        """Load system state from previously processed data"""
        try:
            print("📂 Loading from processed data...")
            
            processed_files = []
            if os.path.exists(self.processed_dir):
                for filename in os.listdir(self.processed_dir):
                    if filename.endswith('_processed.json'):
                        processed_files.append(os.path.join(self.processed_dir, filename))
            
            if not processed_files:
                print("ℹ️ No processed data found")
                return False
            
            # Load processed documents
            import json
            self.processed_documents = []
            
            for filepath in processed_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    processed_doc = json.load(f)
                    self.processed_documents.append(processed_doc)
            
            print(f"✅ Loaded {len(self.processed_documents)} processed documents")
            
            # Re-chunk documents
            print("🔄 Re-chunking documents...")
            self.document_chunks = self.document_chunker.chunk_multiple_documents(self.processed_documents)
            
            # Setup vector database
            await self.setup_vector_database()
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading processed data: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    async def test_assistant():
        assistant = FinancialRAGAssistant()
        
        # Test with sample companies and years
        companies = ['GOOGL', 'MSFT', 'NVDA']
        years = [2023]  # Start with just 2023 for testing
        
        # Extract and process
        success = await assistant.extract_and_process_filings(companies, years)
        if not success:
            print("Failed to extract and process filings")
            return
        
        # Setup vector database
        success = await assistant.setup_vector_database()
        if not success:
            print("Failed to setup vector database")
            return
        
        # Test query
        query = "What was Microsoft's total revenue in 2023?"
        answer = await assistant.answer_query(query)
        print(f"\nAnswer: {answer}")
        
        # Show system status
        status = assistant.get_system_status()
        print(f"\nSystem Status: {status}")
    
    # Run test
    asyncio.run(test_assistant())