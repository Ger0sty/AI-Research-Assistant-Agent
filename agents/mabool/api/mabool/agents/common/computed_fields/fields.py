from __future__ import annotations

import logging
from functools import partial
from typing import Sequence

from ai2i.dcollection import (
    BatchComputedField,
    DocLoadingError,
    Document,
    RelevanceCriteria,
)
from ai2i.di import DI

from mabool.external_api import external_api_deps
from mabool.external_api.rerank.cohere import (
    RerankScoreDocInput,
    RerankScoreInput,
    RerankScorer,
)

logger = logging.getLogger(__name__)


def rerank_score_field(
    relevance_criteria: RelevanceCriteria | None = None, field_name: str = "rerank_score"
) -> BatchComputedField[float]:
    # Don’t require 'markdown' anymore; we’ll compose text inside the loader.
    return BatchComputedField[float](
        field_name=field_name,
        computation_func=partial(_load_rerank_score, relevance_criteria=relevance_criteria),
        required_fields=[],  # no hard requirement; we’ll check fields per-doc
    )


@DI.managed
async def _load_rerank_score(
    entities: Sequence[Document],
    relevance_criteria: RelevanceCriteria | None = None,
    reranker: RerankScorer = DI.requires(external_api_deps.rerank_scorer),
) -> Sequence[float | DocLoadingError]:
    logger.info(f"Adding rerank scores using reranker: {reranker}")

    if not entities:
        return []
    if not relevance_criteria:
        logger.warning("No relevance criteria found. Skipping reranking.")
        return [
            DocLoadingError(corpus_id=d.corpus_id, original_exception=ValueError("No relevance criteria found"))
            for d in entities
        ]

    def _compose_text(d: Document, max_chars: int = 8000) -> str:
        # Prefer markdown if already available (e.g., other pipelines still set it)
        if d.is_loaded("markdown") and d.markdown:
            s = str(d.markdown)
        else:
            # Build from typical SQL fields
            title = (d.title or "").strip() if d.is_loaded("title") else ""
            abstract = (d.abstract or "").strip() if d.is_loaded("abstract") else ""
            body = (d.text or "") if d.is_loaded("text") else ""
            # Simple composition; keep it small for the reranker
            s = "\n\n".join([p for p in [title, abstract, body] if p])
        # Truncate to avoid huge payloads
        return s[:max_chars]

    try:
        docs_payload = []
        order_index = {}
        for idx, document in enumerate(entities):
            txt = _compose_text(document)
            if not txt:
                # If we have no text at all, mark as loading error so lengths stay aligned
                docs_payload.append(None)
            else:
                docs_payload.append(RerankScoreDocInput(corpus_id=document.corpus_id, text=txt))
            order_index[document.corpus_id] = idx

        # Filter out Nones for the API call but keep alignment info
        valid_inputs = [d for d in docs_payload if d is not None]
        if not valid_inputs:
            return [DocLoadingError(corpus_id=d.corpus_id, original_exception=ValueError("No text to rerank"))
                    for d in entities]

        rerank_input = RerankScoreInput(query=relevance_criteria.query, docs=valid_inputs)
        rerank_results = await reranker.rerank(rerank_input=rerank_input)

        # Map results back to original order; missing docs get score 0.0
        scores_by_cid = {r.corpus_id: r.score for r in rerank_results.results}
        out: list[float | DocLoadingError] = []
        for d in entities:
            score = scores_by_cid.get(d.corpus_id)
            if score is None:
                out.append(0.0)
            else:
                out.append(score)
        return out

    except Exception as e:
        logger.exception(f"Failed to rerank using cohere: {e}")
        return [DocLoadingError(corpus_id=d.corpus_id, original_exception=e) for d in entities]


def final_agent_score_field(field_name: str = "final_agent_score") -> BatchComputedField[float]:
    return BatchComputedField[float](
        field_name=field_name,
        cache=False,
        computation_func=_load_final_agent_score,
        required_fields=[
            "broad_search_score",
            "final_specific_paper_by_title_score",
            "final_specific_paper_by_name_score",
        ],
    )


async def _load_final_agent_score(entities: Sequence[Document]) -> Sequence[float | DocLoadingError]:
    agent_score_fields = [
        "broad_search_score",
        "final_specific_paper_by_title_score",
        "final_specific_paper_by_name_score",
    ]
    scores: list[float | DocLoadingError] = []
    for doc in entities:
        loaded_agent_score_fields = [f for f in agent_score_fields if doc.is_loaded(f) and doc[f] is not None]
        if len(loaded_agent_score_fields) == 0:
            logger.warning(f"Document {doc.corpus_id} has no final agent score fields loaded. Setting to 0.")
            score = 0.0
        elif len(loaded_agent_score_fields) > 1:
            logger.warning(
                f"Document {doc.corpus_id} has multiple final agent score fields loaded: {loaded_agent_score_fields}. "
                "Taking the first one."
            )
            score = doc[loaded_agent_score_fields[0]]
        else:
            score = doc[loaded_agent_score_fields[0]]
        scores.append(float(score) if isinstance(score, float) or isinstance(score, int) else 0.0)
    return scores
