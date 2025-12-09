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
from scripts.backend.services.ReRanker.cross_encoder import _cross_encoder_rerank


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
        ce_scores = _cross_encoder_rerank(refined_q, hits)

        for idx, h in enumerate(hits):
            h["cross_encoder"] = ce_scores[idx]

        # FINAL SCORE FUSION
        # vec score ~ semantic
        # bm25 score ~ lexical
        # ce score ~ deep joint relevance
        # tune weights later
        for h in hits:
            vec = h["score"]
            ce = h["cross_encoder"]

            # normalize BM25 for stability

            h["final_score"] = (
                0.25 * vec
                + 0.75 * ce            # cross encoder HEAVILY dominates (expected)
            )

        hits.sort(key=lambda x: x["final_score"], reverse=True)


    # top passage score (for UI)
    top_score: Optional[float] = hits[0]["final_score"] if hits else None

    # ---- 4. Build papers view ----
    papers = build_paper_view(refined_q, hits, analysis)
    top_k_papers = papers[:k]

    # ---- 5. LLM enhancement ----
    enrich_papers_with_llm_explanations(
        user_query=q,
        analysis=analysis,
        papers=top_k_papers,
    )

    # ---- 6. Build combined context ----
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
