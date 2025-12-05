import os
import time
from typing import Optional
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_store: Optional[ElasticsearchStore] = None


def get_es() -> Elasticsearch:
    """Return a plain Elasticsearch client."""
    return Elasticsearch(ES_URL)

def build_store() -> ElasticsearchStore:
    global _store
    if _store is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        es = Elasticsearch(ES_URL)
        _store = ElasticsearchStore(
            index_name=ES_INDEX,
            embedding=embeddings,
            es_connection=es,
            es_url=ES_URL,
            vector_query_field="embedding",
        )
    return _store

def knn_search_with_filters(
    refined_query_vector,
    fetch_k: int,
    filters: list[dict],
):
    """
    Perform a filtered KNN search using Elasticsearch directly.
    This bypasses LangChain so we can use bool.filter queries.
    """
    es = get_es()

    body = {
        "size": fetch_k,
        "query": {
            "bool": {
                "filter": filters,   # authors/venue/year filters
                "must": [
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": refined_query_vector},
                            },
                        }
                    }
                ]
            }
        }
    }

    resp = es.search(index=ES_INDEX, body=body)
    hits = resp["hits"]["hits"]

    # Return in the same format your retriever expects
    out = []
    for h in hits:
        score = h["_score"]
        source = h["_source"]
        out.append((source, score))

    return out

def index_exists_with_docs() -> bool:
    """
    Check whether the index exists and contains at least one document.
    """
    es = get_es()
    try:
        if not es.indices.exists(index=ES_INDEX):
            return False
        count = es.count(index=ES_INDEX).get("count", 0)
        return count > 0
    except Exception:
        return False


def wait_for_index_ready(timeout_s: float = 0.0, poll_s: float = 0.3) -> bool:
    """
    Repeatedly poll until ES index exists and has documents.
    If timeout_s=0, check once.
    """
    if timeout_s <= 0:
        return index_exists_with_docs()

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if index_exists_with_docs():
            return True
        time.sleep(poll_s)

    return index_exists_with_docs()