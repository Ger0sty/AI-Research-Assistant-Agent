from typing import List, Dict, Tuple
from scripts.backend.services.retrieval.elastic_client import build_store
from scripts.backend.services.retrieval.filters import as_author_list, hit_matches_filters
from scripts.backend.services.retrieval.filters import build_filters


def run_vector_search(
    refined_query: str,
    k: int,
    analysis: dict,
    show_scores: bool = True,
) -> List[Dict]:
    """
    Execute vector similarity search using ElasticsearchStore.
    Applies Python-side filtering based on analyzer-derived filters.
    Returns a flat list of chunk hits (not grouped).
    """

    store = build_store()
    filters = build_filters(analysis)

    # Number of chunks to fetch from the vector store.
    # We want more chunks than final papers to allow filtering/grouping.
    # Clamp so we don't go wild.
    fetch_k = max(k * 5, k)   # e.g. k=10 -> fetch_k=50
    fetch_k = min(fetch_k, 500)

    # Call the vector store in a way that *never* directly sets num_candidates,
    # so ES won't see an illegal (num_candidates < k) combo.
    try:
        # Newer LangChain vector stores usually support fetch_k
        results: List[Tuple] = store.similarity_search_with_relevance_scores(
            refined_query,
            k=k,          # how many results we actually want ranked
            fetch_k=fetch_k,  # how many candidates to consider internally
        )
    except TypeError:
        # Older versions without fetch_k: just over-fetch via k itself.
        # LangChain will choose a safe num_candidates >= k internally.
        results: List[Tuple] = store.similarity_search_with_relevance_scores(
            refined_query,
            k=fetch_k,
        )

    hits: List[Dict] = []

    for doc, score in results:
        meta = doc.metadata or {}
        real_score = float(score)

        # Python-side filtering (authors, venues, years)
        if not hit_matches_filters(meta, filters, analysis):
            continue

        hit = {
            "content": doc.page_content,
            "score": real_score,
            "display_score": real_score if show_scores else None,
            "source": meta.get("source"),
            "row": meta.get("row"),
            "start_index": meta.get("start_index"),
            "paper_id": meta.get("paper_id"),
            "title": meta.get("title"),
            "authors": as_author_list(meta.get("authors")),
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": meta.get("url"),
        }

        hits.append(hit)

    return hits