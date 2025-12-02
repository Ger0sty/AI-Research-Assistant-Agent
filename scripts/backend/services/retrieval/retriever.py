from typing import List, Dict, Tuple, Optional
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

    # Retrieve more chunks than needed
    chunk_limit = 10 * k

    results: List[Tuple] = store.similarity_search_with_relevance_scores(
        refined_query,
        k=chunk_limit,
        search_kwargs={"num_candidates": max(chunk_limit, k)},
    )

    hits: List[Dict] = []

    for i, (doc, score) in enumerate(results):
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
