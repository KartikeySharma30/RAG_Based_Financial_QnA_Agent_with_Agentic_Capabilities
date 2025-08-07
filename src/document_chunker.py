"""
Document Chunker
Splits processed documents into chunks based on paragraph separation and semantic boundaries
"""

import re
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    chunk_id: str
    content: str
    metadata: Dict
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    company: str
    year: int
    section: Optional[str] = None

class DocumentChunker:
    """Chunks documents for embedding and retrieval"""
    
    def __init__(self, 
                 max_chunk_size: int = 2000,
                 min_chunk_size: int = 100,
                 overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        
        # Section patterns for 10-K filings
        self.section_patterns = {
            'business': r'item\s+1\s*[.\-–]\s*business',
            'risk_factors': r'item\s+1a\s*[.\-–]\s*risk\s+factors',
            'properties': r'item\s+2\s*[.\-–]\s*properties',
            'legal_proceedings': r'item\s+3\s*[.\-–]\s*legal\s+proceedings',
            'financial_data': r'item\s+6\s*[.\-–]\s*selected\s+financial\s+data',
            'md_a': r'item\s+7\s*[.\-–]\s*management.?s\s+discussion\s+and\s+analysis',
            'financial_statements': r'item\s+8\s*[.\-–]\s*financial\s+statements',
            'controls': r'item\s+9a\s*[.\-–]\s*controls\s+and\s+procedures'
        }
    
    def extract_company_year(self, filepath: str) -> Tuple[str, int]:
        """Extract company ticker and year from filename"""
        filename = filepath.split('/')[-1]
        
        # Pattern: TICKER_10K_YEAR.html
        match = re.search(r'([A-Z]+)_10K_(\d{4})', filename)
        if match:
            return match.group(1), int(match.group(2))
        
        # Fallback
        return "UNKNOWN", 2023
    
    def identify_section(self, text: str) -> Optional[str]:
        """Identify which 10-K section the text belongs to"""
        text_lower = text.lower()
        
        for section_name, pattern in self.section_patterns.items():
            if re.search(pattern, text_lower):
                return section_name
        
        return None
    
    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraph breaks (double newlines)"""
        # First, split by double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 10:  # Filter out very short paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def smart_chunk_text(self, text: str) -> List[str]:
        """Intelligently chunk text considering sentence boundaries and size limits"""
        paragraphs = self.split_by_paragraphs(text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(para) > self.max_chunk_size:
                # Save current chunk if it's not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If paragraph itself is too long, split it further
                if len(para) > self.max_chunk_size:
                    sub_chunks = self._split_long_paragraph(para)
                    chunks.extend(sub_chunks[:-1])  # Add all but last
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk) >= self.min_chunk_size]
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split a long paragraph into smaller chunks at sentence boundaries"""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            overlapped_chunk = chunk
            
            # Add overlap from previous chunk
            if i > 0:
                prev_chunk = chunks[i-1]
                prev_words = prev_chunk.split()
                if len(prev_words) > 20:  # Only add overlap if previous chunk is substantial
                    overlap_text = " ".join(prev_words[-self.overlap_size//10:])
                    overlapped_chunk = overlap_text + "\n\n" + overlapped_chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def generate_chunk_id(self, content: str, source_file: str, index: int) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(source_file.encode()).hexdigest()[:8]
        return f"{file_hash}_{index:04d}_{content_hash}"
    
    def chunk_document(self, processed_doc: Dict) -> List[DocumentChunk]:
        """Chunk a processed document into manageable pieces"""
        filepath = processed_doc['filepath']
        content = processed_doc['processed_content']
        
        # Extract metadata
        company, year = self.extract_company_year(filepath)
        
        # Split into chunks
        chunks = self.smart_chunk_text(content)
        
        # Add overlap
        chunks = self.add_overlap(chunks)
        
        # Create DocumentChunk objects
        document_chunks = []
        start_char = 0
        
        for i, chunk_content in enumerate(chunks):
            # Identify section
            section = self.identify_section(chunk_content)
            
            # Create chunk
            chunk = DocumentChunk(
                chunk_id=self.generate_chunk_id(chunk_content, filepath, i),
                content=chunk_content,
                metadata={
                    'tables_count': len(processed_doc.get('tables', [])),
                    'images_count': len(processed_doc.get('images', [])),
                    'source_type': '10-K',
                    'processing_timestamp': None
                },
                source_file=filepath,
                chunk_index=i,
                start_char=start_char,
                end_char=start_char + len(chunk_content),
                company=company,
                year=year,
                section=section
            )
            
            document_chunks.append(chunk)
            start_char += len(chunk_content)
        
        return document_chunks
    
    def chunk_multiple_documents(self, processed_docs: List[Dict]) -> List[DocumentChunk]:
        """Chunk multiple processed documents"""
        all_chunks = []
        
        for doc in processed_docs:
            print(f"Chunking document: {doc['filepath']}")
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def get_chunk_stats(self, chunks: List[DocumentChunk]) -> Dict:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        company_counts = {}
        year_counts = {}
        section_counts = {}
        
        for chunk in chunks:
            company_counts[chunk.company] = company_counts.get(chunk.company, 0) + 1
            year_counts[chunk.year] = year_counts.get(chunk.year, 0) + 1
            if chunk.section:
                section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'company_distribution': company_counts,
            'year_distribution': year_counts,
            'section_distribution': section_counts
        }

# Example usage and testing
if __name__ == "__main__":
    chunker = DocumentChunker()
    
    # Test with sample processed document
    sample_doc = {
        'filepath': '/workspace/data/sec_filings/GOOGL_10K_2023.html',
        'processed_content': "This is a sample document.\n\nThis is another paragraph with more content.\n\nAnd this is a third paragraph.",
        'tables': [],
        'images': []
    }
    
    chunks = chunker.chunk_document(sample_doc)
    stats = chunker.get_chunk_stats(chunks)
    
    print(f"Created {len(chunks)} chunks")
    print(f"Stats: {stats}")