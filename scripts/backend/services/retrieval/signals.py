from dataclasses import dataclass
from typing import List, Dict, Set, Optional

from scripts.backend.services.retrieval.filters import as_author_list
from scripts.backend.services.retrieval.grouping import norm_query_terms, surname_set


@dataclass
class PaperSignals:
    max_score: float
    mean_score: float
    coverage: int                # number of distinct chunks
    over_threshold: int          # chunks beating a relative threshold
    query_overlap_terms: List[str]
    author_matched: bool
    venue_boost: float
    recency_boost: float


def compute_paper_signals(
    q_terms: Set[str],
    paper_meta: Dict,
    chunks: List[Dict],
    rel_threshold: Optional[float] = None,
) -> PaperSignals:
    """
    Compute scoring signals for a paper based on its chunk hits.
    Matches original behavior from rag_service.py.
    """
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    max_score = max(scores) if scores else 0.0
    mean_score = sum(scores) / len(scores) if scores else 0.0
    coverage = len(chunks)

    thr = (
        rel_threshold
        if rel_threshold is not None
        else (0.8 * max_score if max_score else 0.0)
    )
    over_threshold = sum(1 for s in scores if s is not None and s >= thr)

    # Metadata fields
    title = (paper_meta.get("title") or "").lower()
    authors_list = as_author_list(paper_meta.get("authors"))
    authors_text = " ".join(authors_list).lower()[:200]
    venue = (paper_meta.get("venue") or "").lower()

    # Overlap with chunk text
    text_for_overlap = title + " " + " ".join(
        (c["content"] or "").lower()[:500] for c in chunks
    )
    text_terms = norm_query_terms(text_for_overlap)
    overlap_terms = sorted(q_terms & text_terms)

    # Author signal
    author_matched = bool(surname_set(authors_list) & q_terms)

    # Venue + recency
    venue_boost = 0.15 if any(
        v in venue
        for v in ["acl", "neurips", "iclr", "icml", "emnlp", "naacl", "cvpr", "eccv"]
    ) else 0.0
    year = paper_meta.get("year")
    recency_boost = 0.1 if (isinstance(year, int) and year >= 2022) else 0.0

    return PaperSignals(
        max_score=max_score,
        mean_score=mean_score,
        coverage=coverage,
        over_threshold=over_threshold,
        query_overlap_terms=overlap_terms[:6],
        author_matched=author_matched,
        venue_boost=venue_boost,
        recency_boost=recency_boost,
    )