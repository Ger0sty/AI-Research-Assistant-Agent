# services/pipeline/rag_pipeline.py

import json
from typing import Dict, Any, List, Optional

from scripts.backend.query_analyzer_llm import analyze_query_llm, build_refined_query

from scripts.backend.services.retrieval.retriever import run_vector_search
from scripts.backend.services.llm.explanation_llm import (
    build_paper_view,
    enrich_papers_with_llm_explanations,
)
from scripts.backend.services.ReRanker.bm25 import _bm25_rerank
from scripts.backend.services.ReRanker.final_ranking import final_rank_papers


def query_rag(q: str, k: int = 5, show_scores: bool = True) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    1) Analyze natural-language query
    2) Build refined query
    3) Vector search
    4) Group hits by paper + compute signals
    5) Build per-paper cards + explanations
    6) Optional LLM enriched explanations
    """
    # ---- 1. Analyze query ----
    analysis: dict = analyze_query_llm(q)

    # ---- 2. Build refined text query ----
    refined_q: str = build_refined_query(analysis) or q

    # ---- 3. Retrieve chunk hits ----
    hits = run_vector_search(
        refined_query=refined_q,
        k=k,
        analysis=analysis,
        show_scores=show_scores,
    )

    if hits:
        # BM25 scores per chunk index
        bm25_scores = _bm25_rerank(hits, refined_q)  # {idx: score}

        # Normalize BM25 for stability
        if bm25_scores:
            bm_values = list(bm25_scores.values())
            bm_max = max(bm_values)
            bm_min = min(bm_values)
            denom = (bm_max - bm_min) or 1.0
        else:
            bm_min = 0.0
            denom = 1.0

        for idx, h in enumerate(hits):
            raw_bm25 = bm25_scores.get(idx, 0.0)
            norm_bm25 = (raw_bm25 - bm_min) / denom if denom else 0.0

            h["bm25_score"] = raw_bm25  # raw lexical score for debugging / UI

            vec = float(h.get("score") or 0.0)

            # Final fusion:
            #  - BM25 (lexical) plays the role of ASTA's "relevance judgment score" (dominant)
            #  - Vector score is used as a secondary signal (recall + tie-breaker)
            h["final_score"] = 0.25 * vec + 0.75 * norm_bm25

        hits.sort(key=lambda x: x["final_score"], reverse=True)

    # top passage score (for UI)
    top_score: Optional[float] = hits[0]["final_score"] if hits else None

    # ---- 4. Build papers view ----
    papers = build_paper_view(refined_q, hits, analysis)

    # ---- 5. LLM enhancement ----
    enrich_papers_with_llm_explanations(
        user_query=q,
        analysis=analysis,
        papers=papers,
    )

    # ---- 6. Final ranking ----
    papers = final_rank_papers(papers, analysis)
    top_k_papers = papers[:k]

    # ---- 7. Build combined context ----
    context_parts: List[str] = []
    for p in top_k_papers:
        for c in p["evidence"]:
            context_parts.append(c["content"])

    # ---- Final RAG response ----
    return {
        "query": q,
        "refined_query": refined_q,
        "analysis": analysis,
        "top_score": top_score if show_scores else None,
        "hits": hits,
        "papers": top_k_papers,
        "context": "\n\n---\n\n".join(context_parts),
    }
