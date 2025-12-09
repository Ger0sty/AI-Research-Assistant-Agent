import re

RELEVANCE_ORDER = {
    "perfectly relevant": 3,
    "highly relevant": 2,
    "somewhat relevant": 1,
    "not relevant": 0,
}


def normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def final_rank_papers(papers: list[dict], analysis: dict) -> list[dict]:
    """
    ASTA final deterministic ranking.

    Priority order:
    1. Relevance judgment score (dominant)
    2. Publication date (if recency requested)
    3. Citation count (if centrality requested)
    4. Exact metadata compliance
    5. Exact title / name matching
    """

    recency_pref = analysis.get("recency_preference", False)
    centrality_pref = analysis.get("centrality_preference", False)
    query_title = analysis.get("exact_title") or analysis.get("paper_title")

    norm_query_title = normalize(query_title) if query_title else None

    def sort_key(paper: dict):
        # ---- 1. Relevance (dominant) ----
        relevance = RELEVANCE_ORDER.get(
            paper.get("relevance", "not relevant"),
            0
        )

        # ---- 2. Recency ----
        year = paper.get("year")
        recency = year if (recency_pref and year is not None) else 0

        # ---- 3. Centrality ----
        citations = paper.get("citation_count") or 0
        centrality = citations if centrality_pref else 0

        # ---- 4. Exact metadata compliance ----
        meta = paper.get("metadata_matches", {})
        metadata_score = int(
            bool(meta.get("author")) +
            bool(meta.get("venue")) +
            bool(meta.get("year"))
        )

        # ---- 5. Exact title match ----
        title_match = 0
        if norm_query_title and paper.get("title"):
            pt = normalize(paper["title"])
            if pt == norm_query_title or norm_query_title in pt:
                title_match = 1

        # Lexicographic ordering (higher is better)
        return (
            relevance,
            recency,
            centrality,
            metadata_score,
            title_match,
        )

    return sorted(papers, key=sort_key, reverse=True)
