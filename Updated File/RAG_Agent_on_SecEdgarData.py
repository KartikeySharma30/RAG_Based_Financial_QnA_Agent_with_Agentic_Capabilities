!pip install langchain
!pip install langchain_community
# !pip install tiktoken
# !pip install faiss-cpu
# !pip install chromadb
!pip install edgartools
!pip install langchain-text-splitters python-dotenv
!pip install langchain-google-genai
!pip install -q transformers sentence-transformers qdrant-client langchain
# !pip install pandas numpy pathlib

!pip install edgartools

import os
import pandas as pd
from edgar import set_identity,Company
import numpy as np
import json
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = "Your_API_Key"

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

"""## Data Collection"""

set_identity("K sha kt@codes.finance")

CIK_MAP = {
    "GOOGL": "GOOGL",
    "MSFT": "MSFT",
    "NVDA": "NVDA"
}

YEARS = [2022, 2023, 2024]
SAVE_DIR = "sec_filings"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_10k_for_company(ticker):
    print(f"\nProcessing: {ticker}")
    company = Company(ticker)
    filings = company.get_filings(form="10-K")

    df = filings.to_pandas()
    df['filing_date'] = pd.to_datetime(df['filing_date'])

    for year in YEARS:
        match = df[df['filing_date'].dt.year == year]
        if match.empty:
            print(f"No 10-K filing found for {ticker} in {year}")
            continue

        filing_date = match.iloc[0]['filing_date']
        filing = filings.filter(date=filing_date.strftime('%Y-%m-%d'))
        filing_obj = filing.latest().obj()

        print(f"Downloading {ticker} 10-K for {year} (filed on {filing_date.date()})...")

        try:
            content = filing_obj.items
            filename = f"{ticker}_{year}_10K.txt"
            filepath = os.path.join(SAVE_DIR, filename)

            with open(filepath, 'a', encoding='utf-8') as f:
                for i in content:
                    x = filing_obj[i]
                    f.write(x)

            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Error saving {ticker} {year}: {e}")

if __name__ == "__main__":
    for ticker in CIK_MAP.values():
        download_10k_for_company(ticker)

"""## Chunking , Embedding and Storing"""

from typing import List, Optional

class VectorDatabaseIngestion:
  def __init__(self,
               data_directory: str = "sec_filings/",
               qdrant_url: str = ":memory:",
               collection_name: str = "sec_filings_collection",
               embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
               chunk_size: int = 1000,
               chunk_overlap: int = 200):
    self.data_directory = Path(data_directory)
    self.qdrant_url = qdrant_url
    self.collection_name = collection_name
    self.chunk_size = chunk_size
    self.chunk_overlap = chunk_overlap

    try:
      self.embedding = SentenceTransformerEmbeddings(model_name=embedding_model)
    except Exception as e:
      print(f"Error loading embedding model: {e}")

  def load_documents(self) -> List[Document]:
    """Loads documents from the data directory."""
    try:
      loader = DirectoryLoader(str(self.data_directory),
                              glob="*.txt",
                              loader_cls=TextLoader,
                              show_progress=True)

      documents = loader.load()
      return documents
    except:
      print(f"Error loading documents from {self.data_directory}")
      return []

  def split_documents(self,documents : List[Document]) -> List[Document]:
    """Splits documents into chunks."""

    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(len(chunks))
    return chunks

  def ingest_documents(self):
    documents = self.load_documents()
    if not documents:
      return

    chunks = self.split_documents(documents)
    if not chunks:
      return

    try:
      qdrant = Qdrant.from_documents(
        chunks,
        self.embedding,
        location=self.qdrant_url,
        collection_name=self.collection_name
      )
      self.qdrant_db = qdrant
      return qdrant
    except Exception as e:
      print(f"Error ingesting documents: {e}")


  def search_similar_chunks(self,query: str,top_k: int = 5):
    if not self.qdrant_db:
      print("Qdrant database not initialized. Please call ingest_document first.")
      return []

    try:
      results = self.qdrant_db.similarity_search(query, k=top_k)
      return results
    except Exception as e:
      print(f"Error searching similar chunks: {e}")
      return []

ingester = VectorDatabaseIngestion(
      data_directory=SAVE_DIR,
      qdrant_url=":memory:",
      collection_name="sec_filings_vector_db",
      embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
qdrant_db = ingester.ingest_documents()

"""## Testing Sample Query and Functions"""

query = "What was NVIDIA operating margin in 2023?"
docs = qdrant_db.similarity_search(query)
print("\nSearch Results:\n")
for doc in docs:
    print(doc.page_content[:1000] + "...")
    print('\n'+100*'~'+'\n')

context = "\n\n".join([doc.page_content for doc in docs])

context

Companies = {"GOOGLE":"GOOGL","MICROSOFT":"MSFT","NVIDIA":"NVDA"}
Years = [2022,2023,2024]

prompt = PromptTemplate(
    input_variable = ["context","query","companies","years"],
    template = """ You are a helpful assistant. Use ONLY the following pieces of context provided to answer the question at the end.
    The context might have some data in tabular format so parse and understand it accordingly and answer the question.
    For complex question Like comparsion between companies for revenue/total revenue/margin/operating marging/gross margin/profit earned try to using decomposed the questions provided and then answer based on the context.
    For Simple question You can directly answer the question based on the context.
    If spending/operating margin/gross margin/profit/operating profit/total revenue are not directly given you can calculate them based on the context.
    If the Question ask for revenue growth/growth also provide the percentage growth by calculating it.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Data which we have :

    {context}

    {companies}
    {years}

    Question: {query}
    """
)

LMC = LLMChain(llm=llm,prompt=prompt)
final_a = LMC.run({"context":context,"query":query,"companies":Companies,"years":Years})
print(final_a)

"""### Checking Multi Query Reriever"""

from langchain.retrievers.multi_query import MultiQueryRetriever

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=qdrant_db.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

multiquery_results= multiquery_retriever.invoke(query)

multiquery_results

context = "\n\n".join([doc.page_content for doc in multiquery_results])

LMC = LLMChain(llm=llm,prompt=prompt)
final_a = LMC.run({"context":context,"query":query})
print(final_a)

"""### Checking by Decomposing Query"""

Companies = {"GOOGLE":"GOOGL","MICROSOFT":"MSFT","NVIDIA":"NVDA"}
Years = [2022,2023,2024]

query = "Compare cloud revenue growth rates across all three companies from 2022 to 2023"

decompose_prompt = PromptTemplate(
    input_variables=["companies","years","query"],
    template = """You are a Helpfull assistant. Use ONLY the following pieces of context provided to answer the question at the end.
    I want to decompose this Question/query into multiple simpler and logically ordered sub-queries
    where each Question/query is decomposed on the basis of the question type, company, and year provide to us like this :
    1)Simple Direct Query : "What was Microsoft’s total revenue in 2023?" - For this type of query we don't need to decompose the question as it is already in the best format,So return the question as it is.
    2)Comparative Query : “How did NVIDIA’s data center revenue grow from 2022 to 2023?”
    - it should be broken into Find NVIDIA data center revenue 2022,Find NVIDIA data center revenue 2023,Calculate growth
    3)Cross-Company Analysis : “Which company had the highest operating margin in 2023?”
    - Retrieve MSFT operating margin 2023 , Retrieve GOOGL operating margin 2023,Retrieve NVDA operating margin 2023,Compare and determine highest

    Provide ONLY sub-queries in the above format and place each sub-query into the new line and not in a single line. Not any other text
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {companies}
    {years}
    Question :{query}

    """
)

DLMC = LLMChain(llm=llm,prompt=decompose_prompt)
de_a = DLMC.run({"companies":Companies,"years":Years,"query":query})
print(de_a)

sub_queries = de_a.strip().split('\n')
sub_queries

# Multi step retrieval
all_docs = []
for sub_query in sub_queries:
  sub_docs = multiquery_retriever.invoke(sub_query)
  all_docs.extend(sub_docs)

all_docs

context = "\n\n".join([doc.page_content for doc in all_docs])

context

query = "Compare cloud revenue growth rates across all three companies from 2022 to 2023"

LMC = LLMChain(llm=llm,prompt=prompt)
final_a = LMC.run({"context":context,"query":query,"companies":Companies,"years":Years})
print(final_a)

"""## Agent"""

Companies = {"GOOGLE":"GOOGL","MICROSOFT":"MSFT","NVIDIA":"NVDA"}
Years = [2022,2023,2024]

from langchain.retrievers.multi_query import MultiQueryRetriever

class Agent:
  def __init__(self,qdrant_db):
    self.qdrant_db = qdrant_db

  def decompose_query(self,query):
    decompose_prompt = PromptTemplate(
    input_variables=["companies","years","query"],
    template = """You are a Helpfull assistant. Use ONLY the following pieces of context provided to answer the question at the end.
    I want to decompose this Question/query into multiple simpler and logically ordered sub-queries
    where each Question/query is decomposed on the basis of the question type, company, and year provide to us like this :
    1)Simple Direct Query : "What was Microsoft’s total revenue in 2023?" - For this type of query we don't need to decompose the question as it is already in the best format,So return the question as it is.
    2)Comparative Query : “How did NVIDIA’s data center revenue grow from 2022 to 2023?”
    - it should be broken into Find NVIDIA data center revenue 2022,Find NVIDIA data center revenue 2023,Calculate growth
    3)Cross-Company Analysis : “Which company had the highest operating margin in 2023?”
    - Retrieve MSFT operating margin 2023 , Retrieve GOOGL operating margin 2023,Retrieve NVDA operating margin 2023,Compare and determine highest

    Provide ONLY sub-queries in the above format and place each sub-query into the new line and not in a single line. Not any other text
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {companies}
    {years}
    Question :{query}

    """)

    DLMC = LLMChain(llm=llm,prompt=decompose_prompt)
    de_a = DLMC.run({"companies":Companies,"years":Years,"query":query})

    sub_queries = de_a.strip().split('\n')
    if not sub_queries:
      sub_queries = [query]
    print(sub_queries)
    return sub_queries

  def multistep_retrieval(self,sub_queries):
    multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=qdrant_db.as_retriever(search_kwargs={"k": 5}),
    llm=llm)

    all_docs = []
    for sub_query in sub_queries:
      sub_docs = multiquery_retriever.invoke(sub_query)
      all_docs.extend(sub_docs)

    context = "\n\n".join([doc.page_content for doc in all_docs])
    return context

  def synth_result(self,context,query):
    prompt = PromptTemplate(
    input_variable = ["context","query","companies","years"],
    template = """ You are a helpful assistant. Use ONLY the following pieces of context provided to answer the question at the end.
    The context might have some data in tabular format so parse and understand it accordingly and answer the question.
    For complex question Like comparsion between companies for revenue/total revenue/margin/operating marging/gross margin/profit earned try to using decomposed the questions provided and then answer based on the context.
    For Simple question You can directly answer the question based on the context.
    If spending/operating margin/gross margin/profit/operating profit/total revenue are not directly given you can calculate them based on the context.
    If the Question ask for revenue growth/growth also provide the percentage growth by calculating it.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Data which we have :

    {context}

    {companies}
    {years}

    Question: {query}
    """)

    LMC = LLMChain(llm=llm,prompt=prompt)
    final_a = LMC.run({"context":context,"query":query,"companies":Companies,"years":Years})
    return final_a

  def pipeline(self,query):
    sub_queries = self.decompose_query(query)
    context = self.multistep_retrieval(sub_queries)
    result = self.synth_result(context,query)
    return result

agent = Agent(qdrant_db=qdrant_db)

"""## Testing On Sample Queries"""

query = "How did NVIDIA’s data center revenue grow from 2022 to 2023?"
result = agent.pipeline(query)

print(str(result))

query = "What was NVIDIA's total revenue in fiscal year 2024?"
result = agent.pipeline(query)

print(str(result))

query = "What percentage of Google's 2023 revenue came from advertising?"
result = agent.pipeline(query)

print(str(result))

query = "How much did Microsoft's cloud revenue grow from 2022 to 2023?"
result = agent.pipeline(query)

print(str(result))

query = "Which of the three companies had the highest gross margin in 2023?"
result = agent.pipeline(query)

print(str(result))

query = "Which company had the highest operating margin in 2023?"
result = agent.pipeline(query)

print(str(result))

query = "Compare the R&D spending as a percentage of revenue across all three companies in 2023"
result = agent.pipeline(query)

print(str(result))

query = "What are the main AI risks mentioned by each company and how do they differ?"
result = agent.pipeline(query)

print(str(result))

# Formating and Meta Data can be done based on the requirement
