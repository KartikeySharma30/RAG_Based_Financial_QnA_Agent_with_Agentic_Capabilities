"""
Vector Store with Qdrant
Handles embedding generation, storage, and similarity search
"""

import os
import uuid
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, Range, MatchValue, ScrollResult
)
from .document_chunker import DocumentChunk

class VectorStore:
    """Manages embeddings and vector search with Qdrant"""
    
    def __init__(self, 
                 collection_name: str = "financial_documents",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_api_key: Optional[str] = None):
        
        self.collection_name = collection_name
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            api_key=qdrant_api_key
        )
        
    def create_collection(self) -> bool:
        """Create Qdrant collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                return True
            else:
                print(f"Collection {self.collection_name} already exists")
                return False
                
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            return []
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks with their embeddings"""
        try:
            if not chunks:
                return True
            
            print(f"Storing {len(chunks)} chunks...")
            
            # Generate embeddings for all chunks
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.generate_embeddings(chunk_texts)
            
            if len(embeddings) != len(chunks):
                print("Error: Embedding count doesn't match chunk count")
                return False
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "company": chunk.company,
                        "year": chunk.year,
                        "section": chunk.section,
                        "source_file": chunk.source_file,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        "metadata": chunk.metadata
                    }
                )
                points.append(point)
            
            # Store in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f"  Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
            print(f"Successfully stored {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error storing chunks: {str(e)}")
            return False
    
    def similarity_search(self, 
                         query: str, 
                         top_k: int = 5,
                         company_filter: Optional[str] = None,
                         year_filter: Optional[int] = None,
                         section_filter: Optional[str] = None) -> List[Dict]:
        """Perform similarity search with optional filters"""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Build filters
            filter_conditions = []
            
            if company_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="company",
                        match=MatchValue(value=company_filter)
                    )
                )
            
            if year_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="year",
                        match=MatchValue(value=year_filter)
                    )
                )
            
            if section_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter)
                    )
                )
            
            search_filter = Filter(must=filter_conditions) if filter_conditions else None
            
            # Perform search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    "score": scored_point.score,
                    "chunk_id": scored_point.payload["chunk_id"],
                    "content": scored_point.payload["content"],
                    "company": scored_point.payload["company"],
                    "year": scored_point.payload["year"],
                    "section": scored_point.payload["section"],
                    "source_file": scored_point.payload["source_file"],
                    "metadata": scored_point.payload["metadata"]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []
    
    def multi_query_search(self, 
                          queries: List[str], 
                          top_k_per_query: int = 3,
                          company_filter: Optional[str] = None,
                          year_filter: Optional[int] = None) -> List[Dict]:
        """Perform search for multiple queries and merge results"""
        all_results = []
        seen_chunk_ids = set()
        
        for query in queries:
            results = self.similarity_search(
                query=query,
                top_k=top_k_per_query,
                company_filter=company_filter,
                year_filter=year_filter
            )
            
            # Add unique results
            for result in results:
                if result["chunk_id"] not in seen_chunk_ids:
                    result["query"] = query  # Track which query found this result
                    all_results.append(result)
                    seen_chunk_ids.add(result["chunk_id"])
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get sample of points to analyze
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            
            points = scroll_result[0]
            
            # Analyze distribution
            companies = {}
            years = {}
            sections = {}
            
            for point in points:
                payload = point.payload
                
                company = payload.get("company", "Unknown")
                companies[company] = companies.get(company, 0) + 1
                
                year = payload.get("year", 0)
                years[year] = years.get(year, 0) + 1
                
                section = payload.get("section")
                if section:
                    sections[section] = sections.get(section, 0) + 1
            
            return {
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
                "company_distribution": companies,
                "year_distribution": years,
                "section_distribution": sections
            }
            
        except Exception as e:
            print(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the collection"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore()
    
    # Create collection
    vector_store.create_collection()
    
    # Test with sample chunk
    from .document_chunker import DocumentChunk
    
    sample_chunk = DocumentChunk(
        chunk_id="test_001",
        content="Microsoft Corporation reported total revenue of $198.3 billion in fiscal 2023.",
        metadata={"source_type": "10-K"},
        source_file="test.html",
        chunk_index=0,
        start_char=0,
        end_char=100,
        company="MSFT",
        year=2023,
        section="financial_data"
    )
    
    # Store chunk
    vector_store.store_chunks([sample_chunk])
    
    # Test search
    results = vector_store.similarity_search("Microsoft revenue 2023")
    print(f"Found {len(results)} results")
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")