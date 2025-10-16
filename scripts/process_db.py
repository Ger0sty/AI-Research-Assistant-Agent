import sys
import os
import csv
import shutil
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from dotenv import load_dotenv

# Load paths
ROOT_DIR = Path(__file__).resolve().parents[1]   # one level up from scripts/
load_dotenv(ROOT_DIR / ".env")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_docs")
REBUILD = os.getenv("REBUILD", "0") == "1"

es = Elasticsearch(ES_URL)

# If not forcing rebuild, skip if index exists and has docs
if not REBUILD and es.indices.exists(index=ES_INDEX):
    info = es.count(index=ES_INDEX)
    if info.get("count", 0) > 0:
        print(f"✅ Index '{ES_INDEX}' already has {info['count']} docs — skipping rebuild.")
        raise SystemExit(0)

# Otherwise proceed with your current delete/create logic
if es.indices.exists(index=ES_INDEX):
    print(f"Deleting existing index '{ES_INDEX}'...")
    es.indices.delete(index=ES_INDEX)
print(f"Creating new Elasticsearch index '{ES_INDEX}'...")

# Point of entry
def main():
    documents = load_csv()
    chunks = chunk_docs(documents)
    save_to_elasticsearch(chunks)

# Loads in documents from CSV file
def load_csv():
    try:
        csv.field_size_limit(sys.maxsize)  # often fine on macOS/ARM
    except OverflowError:
        csv.field_size_limit(10**9)        # fallback: 1 GB

    csv_path = ROOT_DIR / "data" / "arxiv_nlp.csv"
    loader = CSVLoader(file_path=str(csv_path))
    documents = loader.load()
    return documents

# Chunks a list of documents
def chunk_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    example = chunks[0]
    print(example.page_content[:300], "...\n", example.metadata)

    return chunks

def save_to_elasticsearch(chunks):
    """
    Save a list of LangChain Document chunks into an Elasticsearch index
    with dense vector embeddings.
    """
    es_url = os.getenv("ES_URL", "http://localhost:9200")
    es_index = os.getenv("ES_INDEX", "rag_docs")
    model_name = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Connect to Elasticsearch
    es_client = Elasticsearch(es_url)

    # Delete existing index (optional for rebuilds)
    try:
        if es_client.indices.exists(index=es_index):
            print(f"Deleting existing index '{es_index}'...")
            es_client.indices.delete(index=es_index)
    except Exception as e:
        print(f"(warning) Could not check/delete index '{es_index}': {e}")

    print(f"Creating new Elasticsearch index '{es_index}'...")

    # Create index with vector field mapping
    # (LangChain does this automatically, but we can still define our own)
    vectorstore = ElasticsearchStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=es_index,
        es_connection=es_client,
        es_url=es_url,
        vector_query_field="embedding",
    )

    print(f"✅ Saved {len(chunks)} chunks to Elasticsearch index '{es_index}'.")


if __name__ == "__main__":
    main()