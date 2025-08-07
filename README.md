# Financial RAG Assistant 🚀

A comprehensive **Retrieval-Augmented Generation (RAG)** system for analyzing SEC financial filings. This system extracts, processes, and enables intelligent querying of 10-K annual filings from **Google (GOOGL)**, **Microsoft (MSFT)**, and **NVIDIA (NVDA)** for years **2022-2024**.

## 🌟 Features

### 📊 **Data Extraction & Processing**
- **SEC Edgar Integration**: Automatically downloads 10-K filings from SEC's Edgar database
- **Multi-Modal Processing**: Handles text, tables, and images from financial documents
- **Table to Markdown**: Converts HTML tables to markdown format for better parsing
- **Image Captioning**: Generates captions for charts and graphs using AI vision models
- **Document Structure Preservation**: Maintains original document structure while enhancing readability

### 🔍 **Intelligent Query Processing**
- **Query Decomposition**: Breaks complex queries into simpler sub-queries
- **Multi-Step Retrieval**: Handles different query types (simple, comparative, cross-company)
- **Smart Filtering**: Company, year, and section-based filtering
- **Calculation Support**: Performs growth calculations and financial comparisons

### 💾 **Vector Database & Search**
- **Qdrant Integration**: High-performance vector similarity search
- **Semantic Chunking**: Intelligent document chunking based on paragraph separation
- **Embedding Storage**: Efficient storage and retrieval of document embeddings
- **Contextual Search**: Advanced similarity search with metadata filtering

### 🤖 **LLM Integration**
- **OpenAI GPT-4**: Advanced language model for answer generation
- **Context Synthesis**: Combines multiple retrieved chunks into coherent context
- **Financial Calculations**: Supports growth rate calculations and comparisons
- **Structured Responses**: Provides detailed, citation-backed answers

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SEC Edgar     │───▶│  Document        │───▶│   Document      │
│   Extractor     │    │  Processor       │    │   Chunker       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Query         │    │  Text + Tables   │    │   Vector Store  │
│   Decomposer    │    │  + Images        │    │   (Qdrant)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                                               │
        ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Multi-Step     │───▶│   Context        │───▶│   LLM Answer    │
│  Retriever      │    │   Synthesis      │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM (for processing large documents)
- Internet connection (for SEC filing downloads)

### Required Services
- **Qdrant Vector Database**: Can run locally or use cloud service
- **OpenAI API**: For GPT-4 and embeddings
- **Optional**: Hugging Face API for custom image captioning

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd financial-rag-assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
# ... other optional keys
```

### 3. Start Qdrant (Local Setup)

```bash
# Using Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant

# Or install locally
# See: https://qdrant.tech/documentation/quick-start/
```

### 4. Run the Assistant

```bash
# Basic usage
python main.py

# Or import in your code
from src import FinancialRAGAssistant

assistant = FinancialRAGAssistant()
# ... use the assistant
```

## 💻 Usage Examples

### Basic Query Processing

```python
import asyncio
from src import FinancialRAGAssistant

async def main():
    # Initialize assistant
    assistant = FinancialRAGAssistant()
    
    # Extract and process filings
    companies = ['GOOGL', 'MSFT', 'NVDA']
    years = [2022, 2023, 2024]
    
    await assistant.extract_and_process_filings(companies, years)
    await assistant.setup_vector_database()
    
    # Ask questions
    queries = [
        "What was Microsoft's total revenue in 2023?",
        "How did NVIDIA's data center revenue grow from 2022 to 2023?",
        "Which company had the highest operating margin in 2023?"
    ]
    
    for query in queries:
        answer = await assistant.answer_query(query)
        print(f"Q: {query}")
        print(f"A: {answer}\n")

asyncio.run(main())
```

### Query Types Supported

#### 1. **Simple Direct Queries**
```python
# Examples:
"What was Microsoft's total revenue in 2023?"
"NVIDIA's data center revenue in 2024"
"Google's operating margin 2022"
```

#### 2. **Comparative Queries** (Temporal Analysis)
```python
# Examples:
"How did NVIDIA's data center revenue grow from 2022 to 2023?"
"Microsoft's revenue change between 2022 and 2024"
"Google's profit growth from 2023 to 2024"
```

#### 3. **Cross-Company Analysis**
```python
# Examples:
"Which company had the highest operating margin in 2023?"
"Compare revenue across all three companies in 2024"
"Who had the best profit margins in 2022?"
```

### Advanced Usage

```python
# Get system status
status = assistant.get_system_status()
print(f"Available companies: {status['available_companies']}")
print(f"Available years: {status['available_years']}")

# Get query examples
examples = assistant.get_query_examples()
for category, queries in examples.items():
    print(f"{category}: {queries}")

# Reset system
await assistant.reset_system()

# Load from processed data (faster startup)
await assistant.load_from_processed_data()
```

## 📁 Project Structure

```
financial-rag-assistant/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── rag_assistant.py         # Main RAG assistant class
│   ├── sec_extractor.py         # SEC Edgar data extraction
│   ├── document_processor.py    # Text, table, image processing
│   ├── document_chunker.py      # Document chunking and splitting
│   ├── vector_store.py          # Qdrant vector database integration
│   ├── query_decomposer.py      # Query analysis and decomposition
│   └── multi_step_retriever.py  # Multi-step retrieval and synthesis
├── data/                        # Raw downloaded data
│   └── sec_filings/            # SEC 10-K HTML files
├── processed_data/              # Processed and structured data
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | ✅ Yes | - |
| `SEC_API_KEY` | SEC API key (optional, for rate limiting) | ❌ No | - |
| `QDRANT_HOST` | Qdrant server host | ❌ No | `localhost` |
| `QDRANT_PORT` | Qdrant server port | ❌ No | `6333` |
| `QDRANT_API_KEY` | Qdrant API key (for cloud) | ❌ No | - |
| `HUGGINGFACE_API_KEY` | Hugging Face API key | ❌ No | - |

### Customization Options

```python
# Initialize with custom settings
assistant = FinancialRAGAssistant(
    data_dir="/custom/data/path",
    processed_dir="/custom/processed/path",
    openai_api_key="your-key",
    qdrant_host="your-qdrant-host",
    qdrant_port=6333
)

# Custom chunking parameters
from src import DocumentChunker
chunker = DocumentChunker(
    max_chunk_size=3000,
    min_chunk_size=200,
    overlap_size=300
)

# Custom vector store settings
from src import VectorStore
vector_store = VectorStore(
    collection_name="custom_financial_docs",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```

## 🎯 Supported Financial Metrics

The system can extract and analyze various financial metrics:

- **Revenue**: Total revenue, net sales, segment revenue
- **Profitability**: Net income, operating income, profit margins
- **Cash Flow**: Operating cash flow, free cash flow
- **Balance Sheet**: Total assets, debt, equity
- **Per-Share Metrics**: Earnings per share (EPS)
- **Segment Data**: Data center revenue, cloud revenue, etc.

## 🔍 Query Processing Pipeline

1. **Query Analysis**: Extract companies, years, and financial metrics
2. **Query Classification**: Categorize as simple, comparative, or cross-company
3. **Query Decomposition**: Break complex queries into sub-queries
4. **Multi-Step Retrieval**: Retrieve relevant chunks for each sub-query
5. **Context Synthesis**: Combine and rank retrieved information
6. **Answer Generation**: Use LLM to generate comprehensive answers

## 🐛 Troubleshooting

### Common Issues

**1. SEC Rate Limiting**
```python
# The system includes built-in rate limiting
# If you encounter 429 errors, increase delays in sec_extractor.py
time.sleep(0.2)  # Increase from 0.1 to 0.2
```

**2. Qdrant Connection Issues**
```bash
# Check if Qdrant is running
curl http://localhost:6333/health

# Restart Qdrant
docker restart <qdrant-container-id>
```

**3. Memory Issues with Large Documents**
```python
# Reduce batch size for processing
batch_size = 50  # Reduce from 100 to 50 in vector_store.py
```

**4. OpenAI API Errors**
```python
# Check API key and quota
import openai
client = openai.OpenAI(api_key="your-key")
print(client.models.list())
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check system status
status = assistant.get_system_status()
print(status)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SEC Edgar Database** for providing public financial data
- **Qdrant** for high-performance vector search
- **OpenAI** for advanced language models
- **Hugging Face** for open-source AI models
- **Financial data community** for supporting open finance tools

## 📞 Support

For questions, issues, or contributions:

- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the example usage patterns

---

**Happy Financial Analysis! 📈💰**
