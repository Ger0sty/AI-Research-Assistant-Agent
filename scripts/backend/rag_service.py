# backend/rag_service.py
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_store = None  # cached across requests

def _build_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    es = Elasticsearch(ES_URL)
    store = ElasticsearchStore(
        index_name=ES_INDEX,
        embedding=embeddings,
        es_connection=es,
        es_url=ES_URL,
        vector_query_field="embedding",
    )
    return store

def get_store():
    global _store
    if _store is None:
        _store = _build_store()
    return _store

def query_rag(query_text: str, k: int = 5):
    store = get_store()
    results = store.similarity_search_with_score(query_text, k=k)  # List[(Document, score)]

    hits = []
    for doc, score in results:
        meta = doc.metadata or {}
        hits.append({
            "content": doc.page_content,
            "score": float(score) if score is not None else None,
            "source": meta.get("source"),
            "row": meta.get("row"),
            "start_index": meta.get("start_index"),
        })

    context_text = "\n\n---\n\n".join([h["content"] for h in hits])
    top_score = hits[0]["score"] if hits else None

    return {
        "query": query_text,
        "top_score": top_score,
        "hits": hits,            # each hit has content + metadata + score
        "context": context_text, # if you want to prompt an LLM later
    }
