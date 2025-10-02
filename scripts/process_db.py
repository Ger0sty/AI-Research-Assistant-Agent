import sys
import os
import csv
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from dotenv import load_dotenv

# Load paths
CHROMA_PATH = "chroma"
ROOT_DIR = Path(__file__).resolve().parents[1]   # one level up from scripts/
load_dotenv(ROOT_DIR / ".env")


# Point of entry
def main():
    documents = load_csv()
    chunks = chunk_docs(documents)
    save_to_chroma(chunks)

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

def save_to_chroma(chunks: list[Document]):
    # Clears database
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # HuggingFace free embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Creates database
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)

    # Save database changes
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()