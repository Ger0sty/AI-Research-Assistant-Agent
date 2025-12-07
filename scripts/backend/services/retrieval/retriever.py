from typing import List, Dict, Tuple
from scripts.backend.services.retrieval.elastic_client import (
    build_store,
    knn_search_with_filters,
)
from scripts.backend.services.retrieval.filters import (
    as_author_list,
    hit_matches_filters,
    build_filters,
)


def run_vector_search(
    refined_query: str,
    k: int,
    analysis: dict,
    show_scores: bool = True,
) -> List[Dict]:

    store = build_store()
    filters = build_filters(analysis)

    # Over-fetch chunks to give grouping + filtering room
    fetch_k = min(max(k * 5, k), 500)

    # ------------------------------------------
    # SEARCH STRATEGY
    # ------------------------------------------
    if filters:
        # ---- Filtered ES-level vector search ----
        embedding = store.embedding.embed_query(refined_query)

        raw_results = knn_search_with_filters(
            refined_query_vector=embedding,
            fetch_k=fetch_k,
            filters=filters,
        )

        # Convert raw ES hits into doc-like objects
        results = []
        for source, score in raw_results:
            doc = type("Doc", (), {})()

            # FIX 1 — use ES "text" field for content
            doc.page_content = (
                source.get("text")
                or source.get("content")
                or ""
            )

            # FIX 2 — unwrap nested metadata
            if "metadata" in source and isinstance(source["metadata"], dict):
                meta = source["metadata"]
            else:
                meta = source

            doc.metadata = meta
            results.append((doc, score))

    else:
        # ---- No filters → plain vector search ----
        results = store.similarity_search_with_relevance_scores(
            refined_query,
            k=k,
            fetch_k=fetch_k,
        )

    # ------------------------------------------
    # PYTHON-SIDE POST-FILTERING
    # ------------------------------------------
    hits: List[Dict] = []

    for doc, score in results:
        raw_meta = doc.metadata or {}

        # Make sure *all* paths (filtered + unfiltered) get unwrapped
        if "metadata" in raw_meta and isinstance(raw_meta["metadata"], dict):
            meta = raw_meta["metadata"]
        else:
            meta = raw_meta

        real_score = float(score)

        # Apply analyzer-derived rules (authors/venue/year/required terms)
        if not hit_matches_filters(meta, filters, analysis, doc.page_content):
            continue

        hits.append({
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
        })

    return hits
