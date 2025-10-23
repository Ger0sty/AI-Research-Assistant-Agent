# backend/rag_service.py
import os
from typing import Dict, Any, List, Tuple, Optional
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

def _build_store() -> ElasticsearchStore:
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

def query_rag(q: str, k: int = 5, show_scores: bool = True) -> Dict[str, Any]:
    """
    Returns:
      {
        "query": str,
        "top_score": float | null,
        "hits": [
          {"content": str, "score": float | null, "source": str | null, "row": int | null, "start_index": int | null},
          ...
        ],
        "context": str
      }
    """
    store = _build_store()
    results: List[Tuple] = store.similarity_search_with_score(q, k=k)

    hits: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    top_score: Optional[float] = None

    for i, (doc, score) in enumerate(results):
        if i == 0 and show_scores:
            top_score = float(score)
        meta = doc.metadata or {}
        hits.append(
            {
                "content": doc.page_content,
                "score": float(score) if show_scores else None,
                "source": meta.get("source"),
                "row": meta.get("row"),
                "start_index": meta.get("start_index"),
            }
        )
        context_parts.append(doc.page_content)

    return {
        "query": q,
        "top_score": top_score if show_scores else None,
        "hits": hits,
        "context": "\n\n---\n\n".join(context_parts),
    }