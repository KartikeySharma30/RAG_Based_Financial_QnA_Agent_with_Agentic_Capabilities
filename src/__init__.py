"""
Financial RAG Assistant
A comprehensive system for extracting, processing, and querying SEC financial filings
"""

from .rag_assistant import FinancialRAGAssistant
from .sec_extractor import SECExtractor
from .document_processor import DocumentProcessor
from .document_chunker import DocumentChunker, DocumentChunk
from .vector_store import VectorStore
from .query_decomposer import QueryDecomposer, QueryType, SubQuery, DecomposedQuery
from .multi_step_retriever import MultiStepRetriever, RetrievalResult, ContextSynthesis

__version__ = "1.0.0"
__author__ = "Financial RAG Assistant Team"

__all__ = [
    "FinancialRAGAssistant",
    "SECExtractor", 
    "DocumentProcessor",
    "DocumentChunker",
    "DocumentChunk",
    "VectorStore",
    "QueryDecomposer",
    "QueryType",
    "SubQuery", 
    "DecomposedQuery",
    "MultiStepRetriever",
    "RetrievalResult",
    "ContextSynthesis"
]