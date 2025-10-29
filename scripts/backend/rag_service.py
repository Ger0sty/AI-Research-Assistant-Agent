# backend/rag_service.py
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch, NotFoundError

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

@dataclass
class PaperSignals:
    max_score: float
    mean_score: float
    coverage: int                # number of distinct chunks
    over_threshold: int          # chunks beating a relative threshold
    query_overlap_terms: list[str]
    author_matched: bool
    venue_boost: float
    recency_boost: float

_store = None  # cached across requests

def index_exists_with_docs(es: Elasticsearch, index: str) -> bool:
    try:
        if not es.indices.exists(index=index):
            return False
        cnt = es.count(index=index).get("count", 0)
        return cnt > 0
    except Exception:
        return False

def wait_for_index_ready(timeout_s: float = 0.0, poll_s: float = 0.3) -> bool:
    """Return True if index has docs, else False. If timeout_s==0, just check once."""
    es = Elasticsearch(ES_URL)
    if timeout_s <= 0:
        return index_exists_with_docs(es, ES_INDEX)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if index_exists_with_docs(es, ES_INDEX):
            return True
        time.sleep(poll_s)
    return index_exists_with_docs(es, ES_INDEX)

def _as_author_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        # ensure stringified
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        # split "A; B, C" robustly
        parts = [p.strip() for p in val.replace(";", ",").split(",")]
        return [p for p in parts if p]
    # fallback for odd types
    return [str(val).strip()] if str(val).strip() else []

def _norm_query_terms(q: str) -> set[str]:
    toks = re.findall(r"[a-z0-9]+", q.lower())
    stop = {"the","a","an","and","or","for","of","to","in","on","with","by","from"}
    return {t for t in toks if t not in stop and len(t) >= 3}

def _group_hits_by_paper(hits: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        pid = h.get("paper_id")
        if not pid:
            src = (h.get("source") or "unknown").split("/")[-1]
            pid = f"{src}::{h.get('title') or 'untitled'}::{h.get('row') or id(h)}"
        grouped[pid].append(h)
    return grouped

def _compute_paper_signals(
    q_terms: set[str],
    paper_meta: dict,
    chunks: list[dict],
    rel_threshold: float | None = None,
) -> PaperSignals:
    scores = [c["score"] for c in chunks if c.get("score") is not None]
    max_score = max(scores) if scores else 0.0
    mean_score = sum(scores)/len(scores) if scores else 0.0
    coverage = len(chunks)
    thr = rel_threshold if rel_threshold is not None else (0.8 * max_score if max_score else 0.0)
    over_threshold = sum(1 for s in scores if s is not None and s >= thr)

    # overlap with title + top authors (very cheap signal)
    title = (paper_meta.get("title") or "").lower()
    authors_list = _as_author_list(paper_meta.get("authors"))
    authors = " ".join(authors_list)[:200].lower()
    venue = (paper_meta.get("venue") or "").lower()

    text_for_overlap = title + " " + " ".join(c["content"].lower()[:500] for c in chunks)
    text_terms = _norm_query_terms(text_for_overlap)
    overlap_terms = sorted(q_terms & text_terms)

    author_matched = any(a.split()[-1] in q_terms for a in paper_meta.get("authors", []) if a)
    # crude boosts you can tune later:
    venue_boost = 0.15 if any(v in venue for v in ["acl","neurips","iclr","icml","emnlp","naacl","cvpr","eccv"]) else 0.0
    year = paper_meta.get("year")
    recency_boost = 0.1 if (isinstance(year, int) and year >= 2022) else 0.0

    return PaperSignals(
        max_score=max_score, mean_score=mean_score, coverage=coverage,
        over_threshold=over_threshold, query_overlap_terms=overlap_terms[:6],
        author_matched=author_matched, venue_boost=venue_boost, recency_boost=recency_boost
    )

def _render_explanation(
    user_query: str,
    paper_meta: dict,
    signals: PaperSignals,
    chunks: list[dict],
    max_snippets: int = 2,
) -> str:
    title = paper_meta.get("title") or "Untitled paper"
    venue = paper_meta.get("venue")
    year = paper_meta.get("year")
    url = paper_meta.get("url")

    # Pick top evidence snippets (by score) and trim for readability
    sorted_chunks = sorted(
        [c for c in chunks if isinstance(c.get("score"), (int,float))],
        key=lambda c: -c["score"]
    ) or chunks
    def _clean(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt.strip())
        # Try to cut at sentence boundary ~240 chars
        return (txt[:240] + "…") if len(txt) > 260 else txt
    snippets = [_clean(c["content"]) for c in sorted_chunks[:max_snippets]]

    lines = []
    # Headline line
    head_bits = []
    if venue and year:
        head_bits.append(f"{venue} {year}")
    elif venue:
        head_bits.append(venue)
    elif isinstance(year, int):
        head_bits.append(str(year))
    if head_bits:
        lines.append(f"Selected from {', '.join(head_bits)}.")
    else:
        lines.append("Selected as a strong candidate.")

    # Why justification
    why_bits = []
    if signals.author_matched:
        why_bits.append("matches the author(s) you mentioned")
    if signals.query_overlap_terms:
        why_bits.append("covers key terms: " + ", ".join(signals.query_overlap_terms))
    if signals.over_threshold >= 2:
        why_bits.append(f"multiple passages scored highly ({signals.over_threshold}+)")
    elif signals.max_score > 0:
        why_bits.append("top passage scored highly")
    if signals.venue_boost > 0:
        why_bits.append("published at a top venue")
    if signals.recency_boost > 0:
        why_bits.append("recent")

    if why_bits:
        lines.append("It was prioritized because it " + "; ".join(why_bits) + ".")

    # Evidence
    if snippets:
        lines.append("Evidence:")
        for snip in snippets:
            lines.append(f"— “{snip}”")

    # Footer link
    if url:
        lines.append(f"Link: {url}")

    return " ".join(lines)

def _extract_paper_meta_from_chunk(chunk: dict) -> dict:
    # Merge chunk metadata into a canonical paper-level dict
    # (Prefer chunk meta, but you could also look up in a separate paper table.)
    return {
        "paper_id": chunk.get("paper_id") or chunk.get("source"),
        "title": chunk.get("title"),
        "authors": _as_author_list(chunk.get("authors")),
        "venue": chunk.get("venue"),
        "year": chunk.get("year"),
        "url": chunk.get("url"),
        "source": chunk.get("source"),
    }

def _build_paper_view(q: str, hits: list[dict]) -> list[dict]:
    q_terms = _norm_query_terms(q)
    by_paper = _group_hits_by_paper(hits)
    papers: list[dict] = []
    # Derive a dynamic threshold using global top score (optional)
    global_max = max((h["score"] for h in hits if isinstance(h.get("score"), (int,float))), default=0.0)
    rel_thr = 0.8 * global_max if global_max else None

    for pid, chunks in by_paper.items():
        # merge paper-level meta
        meta = {}
        for c in chunks:
            meta.update(_extract_paper_meta_from_chunk(c))
        signals = _compute_paper_signals(q_terms, meta, chunks, rel_threshold=rel_thr)
        explanation = _render_explanation(q, meta, signals, chunks, max_snippets=2)

        papers.append({
            "paper_id": pid,
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": meta.get("url"),
            "explanation": explanation,
            "signals": vars(signals),     # handy for UI debugging / sorting
            "evidence": chunks,           # all chunk hits for this paper
        })

    # Sort papers by a composite score (you can tune this)
    def _paper_sort_key(p):
        s = p["signals"]
        return (
            s["max_score"]
            + 0.3 * s["mean_score"]
            + 0.05 * s["coverage"]
            + 0.02 * s["over_threshold"]
            + 0.1 * s["venue_boost"]
            + 0.1 * s["recency_boost"]
        )
    papers.sort(key=_paper_sort_key, reverse=True)
    return papers

def _build_store() -> ElasticsearchStore:
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
    chunk_limit = 10 * k
    results: List[Tuple] = store.similarity_search_with_score(
        q,
        k=chunk_limit,
        search_kwargs={"num_candidates": max(chunk_limit, k)}
    )

    hits: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    top_score: Optional[float] = None

    for i, (doc, score) in enumerate(results):
        real_score = float(score)
        if i == 0:
            top_score = real_score
        meta = doc.metadata or {}
        hit = {
            "content": doc.page_content,
            "score": real_score,
            "display_score": real_score if show_scores else None,
            "source": meta.get("source"),
            "row": meta.get("row"),
            "start_index": meta.get("start_index"),
            # NEW: pass-through paper-level meta if present
            "paper_id": meta.get("paper_id"),
            "title": meta.get("title"),
            "authors": _as_author_list(meta.get("authors")),
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": meta.get("url"),
        }
        hits.append(hit)

    all_papers = _build_paper_view(q, hits)
    top_k_papers = all_papers[:k]

    for p in top_k_papers:
        for c in p["evidence"]:
            context_parts.append(c["content"])

    return {
        "query": q,
        "top_score": top_score if show_scores else None,
        "hits": hits,
        "papers": top_k_papers,
        "context": "\n\n---\n\n".join(context_parts),
    }