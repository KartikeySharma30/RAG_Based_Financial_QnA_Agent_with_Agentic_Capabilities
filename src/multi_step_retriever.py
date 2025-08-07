"""
Multi-Step Retriever
Handles complex query processing through decomposition, retrieval, and synthesis
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import openai
from .query_decomposer import QueryDecomposer, DecomposedQuery, SubQuery, QueryType
from .vector_store import VectorStore

@dataclass
class RetrievalResult:
    """Result from a single retrieval step"""
    sub_query: SubQuery
    retrieved_chunks: List[Dict]
    relevance_scores: List[float]

@dataclass
class ContextSynthesis:
    """Synthesized context from multiple retrieval results"""
    original_query: str
    decomposed_query: DecomposedQuery
    retrieval_results: List[RetrievalResult]
    synthesized_context: str
    requires_calculation: bool
    calculation_data: Optional[Dict] = None

class MultiStepRetriever:
    """Handles complex query processing with multiple retrieval steps"""
    
    def __init__(self, vector_store: VectorStore, openai_api_key: str):
        self.vector_store = vector_store
        self.query_decomposer = QueryDecomposer()
        
        # Initialize OpenAI client
        openai.api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def retrieve_for_sub_query(self, sub_query: SubQuery, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks for a single sub-query"""
        
        # Perform vector search with filters
        search_results = self.vector_store.similarity_search(
            query=sub_query.text,
            top_k=top_k,
            company_filter=sub_query.company,
            year_filter=sub_query.year,
            section_filter=sub_query.section_hint
        )
        
        # Extract relevance scores
        relevance_scores = [result["score"] for result in search_results]
        
        return RetrievalResult(
            sub_query=sub_query,
            retrieved_chunks=search_results,
            relevance_scores=relevance_scores
        )
    
    def retrieve_multi_step(self, decomposed_query: DecomposedQuery) -> List[RetrievalResult]:
        """Retrieve results for all sub-queries"""
        retrieval_results = []
        
        for sub_query in decomposed_query.sub_queries:
            # Skip calculation sub-queries for retrieval
            if sub_query.query_type == QueryType.CALCULATION:
                continue
                
            print(f"Retrieving for: {sub_query.text}")
            result = self.retrieve_for_sub_query(sub_query)
            retrieval_results.append(result)
            print(f"  Found {len(result.retrieved_chunks)} relevant chunks")
        
        return retrieval_results
    
    def extract_financial_data(self, retrieval_results: List[RetrievalResult], decomposed_query: DecomposedQuery) -> Dict:
        """Extract structured financial data for calculations"""
        calculation_data = {}
        
        if not decomposed_query.requires_calculation:
            return calculation_data
        
        # Group results by company and year
        for result in retrieval_results:
            company = result.sub_query.company
            year = result.sub_query.year
            metric = result.sub_query.metric
            
            if company and year:
                key = f"{company}_{year}"
                if key not in calculation_data:
                    calculation_data[key] = {
                        "company": company,
                        "year": year,
                        "chunks": [],
                        "metrics": {}
                    }
                
                calculation_data[key]["chunks"].extend(result.retrieved_chunks)
                
                # Try to extract numerical values from chunks
                for chunk in result.retrieved_chunks:
                    numbers = self._extract_numbers_from_text(chunk["content"], metric)
                    if numbers:
                        if metric not in calculation_data[key]["metrics"]:
                            calculation_data[key]["metrics"][metric] = []
                        calculation_data[key]["metrics"][metric].extend(numbers)
        
        return calculation_data
    
    def _extract_numbers_from_text(self, text: str, metric: Optional[str] = None) -> List[Dict]:
        """Extract numerical values from text with context"""
        import re
        
        numbers = []
        
        # Pattern for financial numbers (billions, millions, etc.)
        pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand|B|M|K)?'
        
        matches = re.finditer(pattern, text, re.IGNORECASE)
        
        for match in matches:
            value_str = match.group(1).replace(',', '')
            unit = match.group(2)
            
            try:
                value = float(value_str)
                
                # Convert to base units (dollars)
                if unit:
                    unit_lower = unit.lower()
                    if unit_lower in ['billion', 'b']:
                        value *= 1_000_000_000
                    elif unit_lower in ['million', 'm']:
                        value *= 1_000_000
                    elif unit_lower in ['thousand', 'k']:
                        value *= 1_000
                
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                numbers.append({
                    "value": value,
                    "original_text": match.group(0),
                    "context": context,
                    "unit": unit
                })
                
            except ValueError:
                continue
        
        return numbers
    
    def synthesize_context(self, retrieval_results: List[RetrievalResult], decomposed_query: DecomposedQuery) -> str:
        """Synthesize retrieved chunks into coherent context"""
        
        # Group chunks by relevance and remove duplicates
        all_chunks = []
        seen_chunk_ids = set()
        
        for result in retrieval_results:
            for chunk in result.retrieved_chunks:
                if chunk["chunk_id"] not in seen_chunk_ids:
                    all_chunks.append({
                        **chunk,
                        "sub_query": result.sub_query.text
                    })
                    seen_chunk_ids.add(chunk["chunk_id"])
        
        # Sort by relevance score
        all_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        # Build synthesized context
        context_parts = []
        context_parts.append(f"Query: {decomposed_query.original_query}")
        context_parts.append(f"Query Type: {decomposed_query.query_type.value}")
        context_parts.append("")
        
        # Add relevant information
        for i, chunk in enumerate(all_chunks[:10]):  # Limit to top 10 chunks
            company = chunk["company"]
            year = chunk["year"]
            section = chunk["section"] or "General"
            
            context_parts.append(f"### Relevant Information {i+1}")
            context_parts.append(f"**Source**: {company} {year} ({section})")
            context_parts.append(f"**Relevance**: {chunk['score']:.3f}")
            context_parts.append(f"**Content**: {chunk['content'][:1000]}...")  # Truncate if too long
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def process_query(self, query: str) -> ContextSynthesis:
        """Main method to process a complex query end-to-end"""
        
        print(f"Processing query: {query}")
        
        # Step 1: Decompose query
        print("Step 1: Decomposing query...")
        decomposed_query = self.query_decomposer.decompose_query(query)
        print(f"  Query type: {decomposed_query.query_type.value}")
        print(f"  Sub-queries: {len(decomposed_query.sub_queries)}")
        
        # Step 2: Multi-step retrieval
        print("Step 2: Multi-step retrieval...")
        retrieval_results = self.retrieve_multi_step(decomposed_query)
        
        # Step 3: Extract calculation data if needed
        calculation_data = None
        if decomposed_query.requires_calculation:
            print("Step 3: Extracting calculation data...")
            calculation_data = self.extract_financial_data(retrieval_results, decomposed_query)
        
        # Step 4: Synthesize context
        print("Step 4: Synthesizing context...")
        synthesized_context = self.synthesize_context(retrieval_results, decomposed_query)
        
        return ContextSynthesis(
            original_query=query,
            decomposed_query=decomposed_query,
            retrieval_results=retrieval_results,
            synthesized_context=synthesized_context,
            requires_calculation=decomposed_query.requires_calculation,
            calculation_data=calculation_data
        )
    
    def generate_final_answer(self, context_synthesis: ContextSynthesis) -> str:
        """Generate final answer using LLM with synthesized context"""
        
        # Prepare system prompt
        system_prompt = """You are a financial analyst assistant. Your task is to answer questions about financial data from SEC 10-K filings.

Key guidelines:
1. Base your answers strictly on the provided context
2. When performing calculations, show your work step by step
3. Include specific numbers with units (millions, billions)
4. Cite the relevant company and year for data points
5. If you cannot find specific information, state this clearly
6. For comparative queries, present data in a structured format

The context includes relevant excerpts from financial documents. Use this information to provide accurate, detailed answers."""

        # Prepare user prompt
        user_prompt = f"""Please answer the following financial question based on the provided context:

Question: {context_synthesis.original_query}

Context:
{context_synthesis.synthesized_context}

Additional Information:
- Query requires calculation: {context_synthesis.requires_calculation}
- Query type: {context_synthesis.decomposed_query.query_type.value}

Please provide a comprehensive answer based on the available information."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    # This would require proper setup with vector store and API keys
    print("MultiStepRetriever component ready for integration")