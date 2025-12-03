# backend/rag_service.py
import os
import re
import time
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch, NotFoundError
from scripts.backend.query_analyzer_llm import (
    analyze_query_llm,
    build_refined_query,
)
from scripts.backend.llm_utils import call_llm_json
from rank_bm25 import BM25Okapi
import nltk
from sentence_transformers import CrossEncoder



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

cross_encoder = None

def _load_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return cross_encoder


# build ES filters
def _build_filters(analysis: dict) -> list[dict]:
    filters: list[dict] = []
    authors = analysis.get("authors") or []
    venues  = analysis.get("venues") or []
    yr      = (analysis.get("time_range") or {}) if isinstance(analysis.get("time_range"), dict) else {}

    if authors:
        filters.append({"terms": {"authors.keyword": authors}})
    if venues:
        filters.append({"terms": {"venue.keyword": [v.upper() for v in venues]}})
    start, end = yr.get("start"), yr.get("end")
    if start or end:
        rng = {"gte": start} if start else {}
        if end: rng["lte"] = end
        filters.append({"range": {"year": rng}})
    return filters


def _bm25_rerank(chunks: list[dict], query: str) -> dict[str, float]:
    """
    Return a dict mapping each chunk-id → BM25 score.
    chunk-id = index in list.
    """
    tokenized_docs = []
    for c in chunks:
        text = (c["content"] or "").lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        tokenized_docs.append(tokens)

    bm25 = BM25Okapi(tokenized_docs)
    q_tokens = re.findall(r"[a-z0-9]+", query.lower())
    scores = bm25.get_scores(q_tokens)

    # Return {chunk_index: bm25_score}
    return {i: float(scores[i]) for i in range(len(chunks))}

def _cross_encoder_rerank(query: str, hits: list[dict]) -> dict[int, float]:
    """
    Returns: {chunk_index: cross_encoder_score}
    """
    if not hits:
        return {}
    
    ce = _load_cross_encoder()

    # Build sentence pairs: (query, chunk_content)
    pairs = [(query, h["content"]) for h in hits]

    # CrossEncoder returns relevance scores
    ce_scores = ce.predict(pairs, batch_size=8, show_progress_bar=False)


    return {i: float(ce_scores[i]) for i in range(len(hits))}


def _search_by_author(es, refined_q: str, filters: list[dict], k: int):
    must = [{"match": {"content": refined_q}}] if refined_q else []
    body = {"query": {"bool": {"must": must, "filter": filters}}, "size": k}
    return es.search(index=ES_INDEX, body=body)

def _search_metadata_only(es, refined_q: str, filters: list[dict], k: int):
    must = [{"match": {"title": refined_q}}] if refined_q else [{"match_all": {}}]
    body = {"query": {"bool": {"must": must, "filter": filters}}, "size": k}
    return es.search(index=ES_INDEX, body=body)

def _search_broad(es, refined_q: str, filters: list[dict], k: int):
    must = [{"multi_match": {"query": refined_q, "fields": ["title^3", "content"]}}] if refined_q else [{"match_all": {}}]
    body = {"query": {"bool": {"must": must, "filter": filters}}, "size": k}
    return es.search(index=ES_INDEX, body=body)

def _search_specific_title(es, refined_q: str, filters: list[dict], k: int):
    must = [{"match_phrase": {"title": refined_q}}] if refined_q else [{"match_all": {}}]
    body = {"query": {"bool": {"must": must, "filter": filters}}, "size": k}
    return es.search(index=ES_INDEX, body=body)

_explain_cache: dict[str, dict] = {}

def _name_tokens(n: str) -> list[str]:
    return re.findall(r"[a-z]+", (n or "").lower())

def _surname_set(names: list[str]) -> set[str]:
    out = set()
    for n in names or []:
        toks = _name_tokens(n)
        if toks:
            out.add(toks[-1])
    return out

def _paper_cache_key(meta: dict, refined_q: str) -> str:
    pid = str(meta.get("paper_id") or meta.get("title") or "")[:120]
    y   = meta.get("year")
    return f"{refined_q}::{pid}::{y}"

def _abstract_from_chunks(chunks: list[dict], max_chars: int = 1200) -> str:
    """Prefer any chunk that looks like an abstract; else take the top-scored text."""
    # heuristic: if a chunk’s source/title mentions 'abstract' or it’s the first chunk
    best = None
    for c in sorted(chunks, key=lambda c: float(c.get("score") or 0.0), reverse=True):
        t = (c.get("content") or "")
        if not best:
            best = t
        if re.search(r"\babstract\b", (c.get("source") or "") + " " + (c.get("title") or ""), flags=re.I):
            best = t
            break
    best = re.sub(r"\s+", " ", (best or "").strip())
    return best[:max_chars]

# NEW — the LLM rewriter (sync; uses your call_llm_json)
def llm_explain_choice(meta: dict, analysis: dict, chunks: list[dict],
                       signals: PaperSignals, refined_q: str,
                       max_quotes: int = 2) -> dict:
    """
    Returns:
      {
        "why": str,              # 2-3 sentences, concrete
        "bullets": [str, ...],   # up to 4 crisp reasons
        "evidence": [str, ...],  # 0-2 short quotes (<=140 chars)
        "score_note": str        # tiny note like 'author match + recent'
      }
    """
    key = _paper_cache_key(meta, refined_q)
    if key in _explain_cache:
        return _explain_cache[key]

    abstract = _abstract_from_chunks(chunks, 1200)
    short_snips = []
    for c in sorted(chunks, key=lambda c: float(c.get("score") or 0.0), reverse=True)[:max_quotes]:
        t = re.sub(r"\s+", " ", (c.get("content") or "").strip())
        if t:
            short_snips.append(t[:140])

    prompt = {
        "task": "justify_selection",
        "constraints": {
            "max_chars": 600,
            "avoid_generic": ["paper", "study", "approach", "method", "model"],
            "require_concrete_terms": True,
            "min_specific_reasons": 2,
        },
        "query": {
            "original": analysis.get("original_query", "") or "",   # may not exist; ok
            "refined": refined_q,
            "authors": analysis.get("authors") or [],
            "venues": analysis.get("venues") or [],
            "time_range": analysis.get("time_range") or {},
            "query_type": (analysis.get("query_type") or {}).get("type", "")
        },
        "paper": {
            "title": meta.get("title"),
            "authors": meta.get("authors") or [],
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": meta.get("url")
        },
        "signals": {
            "overlap_terms": signals.query_overlap_terms,
            "author_matched": signals.author_matched,
            "venue_boost": signals.venue_boost,
            "recency_boost": signals.recency_boost,
            "over_threshold": signals.over_threshold
        },
        "abstract": abstract,
        "evidence_snippets": short_snips
    }

    schema = {
        "why": "string (2-3 sentences, concrete, < 360 chars, no filler)",
        "bullets": ["string (max 4; each < 90 chars; specific)"],
        "evidence": ["string (<= 2 quotes; each <= 140 chars; optional)"],
        "score_note": "string (<= 40 chars; optional)"
    }

    # Ask the LLM for strict JSON; your wrapper should return a dict or {"error": "..."}
    generic_words = ["paper", "study", "approach", "method", "model"]
    resp = call_llm_json(
        f"Return ONLY JSON matching this schema:\n{schema}\n\n"
        f"Be specific: mention at least one concept from overlap_terms or abstract nouns. "
        f"If author filters matched, say so. If venue/year requested and satisfied, say so. "
        f"Avoid generic words: {generic_words}.\n\n"
        f"INPUT:\n{prompt}",
        max_new_tokens=256
    )

    # Fallback if anything goes wrong
    if not isinstance(resp, dict) or "why" not in resp:
        resp = {
            "why": _compose_one_liner(meta, signals, analysis, refined_q),
            "bullets": [],
            "evidence": short_snips[:max_quotes],
            "score_note": ""
        }

    _explain_cache[key] = resp
    return resp

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
    analysis, 
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

    author_matched = bool(_surname_set(authors_list) & q_terms)
    # crude boosts you can tune later:
    venue_boost = 0.15 if any(v in venue for v in ["acl","neurips","iclr","icml","emnlp","naacl","cvpr","eccv"]) else 0.0
    year = paper_meta.get("year")
    recency_boost = 0.1 if (isinstance(year, int) and year >= 2022) else 0.0
    pref = analysis.get("recency") if isinstance(analysis, dict) else None
    if pref == "recent":
        recency_boost = 0.2 if year >= 2023 else 0
    elif pref == "early":
        recency_boost = 0.2 if year <= 2018 else 0

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


_POSITIVE_EVAL = r"\b(experiment|evaluation|evaluat(e|ion)|benchmark|results|baseline|ablation)\b"
_DATASET_HINT  = r"\b(dataset|corpus|collection|annotations?|labeled|release|we (introduce|present|release))\b"
_METHOD_HINT   = r"\b(approach|method|model|framework|algorithm|pipeline)\b"

def _has_keyword(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def _extract_short_fact(text: str, keywords: list[str], max_len: int = 180) -> Optional[str]:
    """
    Return one short sentence that contains any keyword; keep it concise.
    """
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    keys = [k.lower() for k in keywords if k]
    for s in sents:
        low = s.lower()
        if any(k in low for k in keys):
            s = re.sub(r"\s+", " ", s).strip()
            return (s[:max_len] + "…") if len(s) > max_len else s
    return None

def _grade_relevance(signals: PaperSignals, meta: dict, analysis: dict) -> tuple[str, float]:
    """
    Return (label, score_in_[0,1]).
    """
    score = 0.0
    # retrieval quality
    score += 0.45 * (1.0 / (1.0 + math.exp(-((signals.max_score or 0) - 0.5) * 2.0)))
    score += 0.15 * min(1.0, (signals.over_threshold or 0) / 3.0)
    score += 0.10 * (0.2 if signals.author_matched else 0.0)
    score += 0.10 * signals.venue_boost
    score += 0.05 * signals.recency_boost

    # analyzer alignment: authors/venue/year present = small bump
    if analysis.get("authors"): score += 0.04
    if analysis.get("venues"):  score += 0.04
    if (analysis.get("time_range") or {}).get("start") or (analysis.get("time_range") or {}).get("end"):
        score += 0.02

    score = max(0.0, min(1.0, score))
    if score >= 0.75: label = "Perfectly Relevant"
    elif score >= 0.50: label = "Relevant"
    else: label = "Somewhat Relevant"
    centr = analysis.get("centrality")
    if centr == "first":
        score += 0.08
    elif centr == "last":
        score -= 0.08

    bos = analysis.get("broad_or_specific")
    if bos == "unique-identifier":
        score += 0.05
    elif bos == "descriptions-or-keywords":
        score += 0.02


    return label, score

def _compose_card(meta: dict, signals: PaperSignals, analysis: dict, chunks: list[dict]) -> dict:
    """
    Build Asta-like justification WITHOUT pasting long chunks.
    Produces: verdict, justification, tags, facts (short), url.
    """
    # gather lightweight cues from top chunk text
    top = max(chunks, key=lambda c: float(c.get("score") or 0.0)) if chunks else {}
    text = (top.get("content") or "")[:2500]  # small budget

    # tags (boolean cues → badges)
    tags = set()

    bos = analysis.get("broad_or_specific")
    if bos == "unique-identifier":
        tags.add("Specific Query")
    elif bos == "descriptions-or-keywords":
        tags.add("Broad Query")


    # Focus tags from analyzer keywords intersecting with title/content
    q_keywords = [k for k in (analysis.get("keywords") or []) if isinstance(k, str)]
    focus_hit = _extract_short_fact(text, q_keywords) if q_keywords else None
    if q_keywords:
        tags.add("Query Focus")

    # empirical evaluation?
    if _has_keyword(text, _POSITIVE_EVAL):
        tags.add("Empirical Evaluation")

    # dataset/corpus?
    if _has_keyword(text, _DATASET_HINT):
        tags.add("Dataset/Corpus")

    # method/model?
    if _has_keyword(text, _METHOD_HINT):
        tags.add("Method/Model")

    # venue/year recency tags
    if signals.recency_boost > 0:
        tags.add("Recent")

    venue = (meta.get("venue") or "").strip()
    if venue:
        tags.add(venue.upper()[:18])

    # author match
    if signals.author_matched:
        tags.add("Author Match")

    # verdict & justification (one tidy sentence)
    label, rel = _grade_relevance(signals, meta, analysis)
    pieces = [f"{label}:"]
    why = []

    if signals.author_matched:
        why.append("matches requested author(s)")
    if q_keywords:
        why.append("focuses on your keywords")
    if "Empirical Evaluation" in tags:
        why.append("includes an empirical evaluation")
    if "Dataset/Corpus" in tags:
        why.append("releases a dataset/corpus")
    if venue:
        y = meta.get("year")
        why.append(f"published at {venue}{' ' + str(y) if isinstance(y, int) else ''}")
    elif isinstance(meta.get("year"), int):
        why.append(f"published in {meta['year']}")

    if signals.venue_boost > 0:
        why.append("top-tier venue")
    if signals.recency_boost > 0 and "Recent" not in tags:
        why.append("recent")

    if signals.over_threshold >= 2:
        why.append("multiple passages scored highly")

    if not why:
        why.append("high retrieval score")

    justification = " ".join(pieces) + " " + "; ".join(why) + "."
    # short fact(s)—at most one sentence, trimmed
    facts = []
    if focus_hit:
        facts.append(focus_hit)
    else:
        # generic fact pickers
        for kws in [["dataset", "corpus"], ["experiment", "baseline", "results"]]:
            s = _extract_short_fact(text, kws)
            if s and s not in facts:
                facts.append(s)
            if len(facts) >= 2:
                break

    return {
        "verdict": label,
        "score": rel,
        "justification": justification,
        "tags": sorted(tags),
        "facts": facts,            # 0–2 short sentences, not raw big chunks
        "url": meta.get("url"),
    }


def _compose_explanation(paper_meta: dict, analysis: dict, signals: PaperSignals, chunks: list[dict]) -> str:
    """
    Turn analyzer fields + paper metadata into a short, human-readable reason.
    Prefer analyzer filters (authors/venues/years) and include 1–2 evidence snippets.
    """
    title = paper_meta.get("title") or "Untitled paper"
    venue = paper_meta.get("venue")
    year = paper_meta.get("year")
    url  = paper_meta.get("url")

    asked_authors = analysis.get("authors") or []
    asked_venues  = analysis.get("venues") or []
    yr = analysis.get("time_range") or {}
    asked_range = (yr.get("start"), yr.get("end"))

    bits = []

    # Header
    if venue and year:
        bits.append(f"Selected from {venue} {year}.")
    elif venue:
        bits.append(f"Selected from {venue}.")
    elif isinstance(year, int):
        bits.append(f"Selected ({year}).")
    else:
        bits.append("Selected as a strong candidate.")

    # Why this paper
    why = []
    if asked_authors and any(a in (paper_meta.get("authors") or []) for a in asked_authors):
        why.append("matches requested author(s)")
    if asked_venues and venue and venue.upper() in {v.upper() for v in asked_venues}:
        why.append("published at a requested venue")
    if asked_range and any(asked_range):
        s, e = asked_range
        if isinstance(year, int) and ((s is None or year >= s) and (e is None or year <= e)):
            why.append("falls in requested year range")
    if signals.query_overlap_terms:
        why.append("covers key terms: " + ", ".join(signals.query_overlap_terms))
    if signals.over_threshold >= 2:
        why.append(f"has multiple highly-scored passages ({signals.over_threshold}+)")
    elif signals.max_score > 0:
        why.append("contains a highly-scored passage")
    if signals.venue_boost > 0:
        why.append("strong venue")
    if signals.recency_boost > 0:
        why.append("recent")

    if why:
        bits.append("It was prioritized because it " + "; ".join(why) + ".")

    # Evidence (top 1–2 short snippets)
    top = sorted([c for c in chunks if isinstance(c.get("score"), (int, float))],
                 key=lambda c: -c["score"]) or chunks
    def short(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt.strip())
        return (txt[:240] + "…") if len(txt) > 260 else txt
    snips = [short(c["content"]) for c in top[:2]]
    if snips:
        bits.append("Evidence:")
        for s in snips:
            bits.append(f"— “{s}”")

    if url:
        bits.append(f"Link: {url}")

    return " ".join(bits)

def _filters_from_analysis(analysis: dict) -> dict:
    f: dict = {}
    authors = analysis.get("authors") or []
    venues  = analysis.get("venues") or []
    yr      = analysis.get("time_range") or {}

    if authors:
        f["authors"] = [a.strip() for a in authors if isinstance(a, str) and a.strip()]
    if venues:
        f["venues"]  = [v.strip().lower() for v in venues if isinstance(v, str) and v.strip()]
    if isinstance(yr, dict) and (yr.get("start") or yr.get("end")):
        f["year_range"] = (yr.get("start"), yr.get("end"))
    return f

def _hit_matches_filters(hit: dict, f: dict) -> bool:
    if not f:
        return True

    # authors (any overlap)
    want_authors = set(a.casefold() for a in f.get("authors", []))
    if want_authors:
        hit_authors = set(a.casefold() for a in (hit.get("authors") or []) if isinstance(a, str))
        if not (want_authors & hit_authors):
            return False

    # venues (case-insensitive exact token match)
    want_venues = set(f.get("venues", []))
    if want_venues:
        v = hit.get("venue")
        if not (isinstance(v, str) and v.strip().lower() in want_venues):
            return False

    # year range (inclusive)
    if "year_range" in f:
        start, end = f["year_range"]
        y = hit.get("year")
        if isinstance(y, int):
            if start is not None and y < start:
                return False
            if end   is not None and y > end:
                return False
        else:
            # if we require a range but hit has no year, reject
            return False

    return True

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

def _compose_one_liner(meta: dict, signals: PaperSignals, analysis: dict, refined_q: str) -> str:
    """
    Deterministic per-paper summary using topic, author match, overlap terms, and venue/year.
    """
    topic = (analysis.get("refined_query") or refined_q or "").strip()
    if not topic:
        kws = analysis.get("keywords") or []
        if isinstance(kws, list) and kws:
            topic = ", ".join(kws[:3])

    bits: list[str] = []

    # Always state topic relevance
    if topic:
        bits.append(f'is relevant to "{topic}"')

    # Add paper-specific overlap terms (top 2) to make sentences differ
    if signals.query_overlap_terms:
        terms = ", ".join(signals.query_overlap_terms[:2])
        bits.append(f"covers key terms: {terms}")

    # Author match (only when it actually matches)
    want = {a.strip().casefold() for a in (analysis.get("authors") or [])}
    have = {a.strip().casefold() for a in (meta.get("authors") or [])}
    overlap = sorted(want & have)
    if overlap:
        named = ", ".join(o.title() for o in overlap[:2])
        bits.append("matches the requested author(s)" + (f": {named}" if named else ""))

    # Retrieval strength
    if signals.over_threshold >= 2:
        bits.append("has multiple highly-scored passages")
    elif signals.max_score > 0:
        bits.append("has a strong top passage")

    # Quality hints
    if signals.venue_boost > 0:
        bits.append("comes from a top-tier venue")
    if signals.recency_boost > 0:
        bits.append("is recent")

    # Build the sentence
    if len(bits) >= 2:
        body = "This paper was prioritized because it " + "; ".join(bits[:-1]) + f"; and {bits[-1]}."
    elif bits:
        body = "This paper was prioritized because it " + bits[0] + "."
    else:
        body = "This paper was prioritized because it scored highly."

    # Tail: venue/year if available (helps sentences differ)
    venue = meta.get("venue")
    year  = meta.get("year")
    tail = " ".join(x for x in [venue, str(year) if isinstance(year, int) else None] if x)
    if tail:
        body += f" Published at {tail}."

    return body

def _build_paper_view(q: str, hits: list[dict], analysis: dict) -> list[dict]:
    q_terms = _norm_query_terms(q)
    by_paper = _group_hits_by_paper(hits)
    papers: list[dict] = []
    global_max = max((h["score"] for h in hits if isinstance(h.get("score"), (int,float))), default=0.0)
    rel_thr = 0.8 * global_max if global_max else None

    for pid, chunks in by_paper.items():
        meta = {}
        for c in chunks:
            meta.update(_extract_paper_meta_from_chunk(c))

        signals = _compute_paper_signals(q_terms, meta, chunks, analysis, rel_threshold=rel_thr)

        # build the Asta-style card
        card = _compose_card(meta, signals, analysis, chunks)

        # overwrite (or set) the one-liner justification
        one_liner = _compose_one_liner(meta, signals, analysis, q)

        # LLM justification
        llm_just = llm_explain_choice(meta, analysis, chunks, signals, q)
        card["justification"] = llm_just.get("why") or one_liner
        card["bullets"] = llm_just.get("bullets", [])
        card["evidence_quotes"] = llm_just.get("evidence", [])
        card["score_note"] = llm_just.get("score_note", "")

        papers.append({
            "paper_id": pid,
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": meta.get("url"),
            "card": card,
            "explanation": card["justification"], 
            "signals": vars(signals),
            "evidence": chunks,
        })

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
    analysis: dict = analyze_query_llm(q)
    refined_q: str = build_refined_query(analysis) or q
    analysis["refined_query"] = refined_q

    filters = _build_filters(analysis)

    # Create Elasticsearch client
    es = Elasticsearch(ES_URL)

    # Decide search mode based on "by_title_or_name"
    qt = analysis.get("by_title_or_name")

    bos = analysis.get("broad_or_specific")
    if bos == "unique-identifier":
        qt = "title"
    elif bos == "descriptions-or-keywords":
        qt = "name"

    if qt == "title":
        # Strong exact title search
        raw_es = _search_specific_title(es, refined_q, filters, k)
    elif qt == "name":
        # Keyword / semantic style
        raw_es = _search_broad(es, refined_q, filters, k)
    else:
        # Default fallback → same as your current code
        raw_es = _search_broad(es, refined_q, filters, k)


    # lightweight Python-side filter using analyzer fields
    def _passes_filters(meta: dict) -> bool:
        # authors
        if any(f.get("terms", {}).get("authors.keyword") for f in filters):
            want = {a.lower() for a in (analysis.get("authors") or [])}
            have = {a.lower() for a in _as_author_list(meta.get("authors"))}
            if want and want.isdisjoint(have):
                return False
        # venues
        ven_terms = next(
            (set(v.lower() for v in f["terms"]["venue.keyword"])
             for f in filters if "terms" in f and "venue.keyword" in f["terms"]),
            None
        )
        if ven_terms:
            venue = str(meta.get("venue") or "").lower()
            if venue and venue.lower() not in ven_terms:
                # allow partial match (ACL 2024 vs ACL)
                if not any(v in venue for v in ven_terms):
                    return False
        # years
        rng = next((f["range"]["year"]
                    for f in filters if "range" in f and "year" in f["range"]), None)
        if rng:
            y = meta.get("year")
            if isinstance(y, int):
                if "gte" in rng and y < rng["gte"]: return False
                if "lte" in rng and y > rng["lte"]: return False
        return True

    # --- existing vector search, but use refined_q instead of q ---
    store = _build_store()
    chunk_limit = 10 * k
    results: List[Tuple] = store.similarity_search_with_score(
        refined_q,                                # << use analyzer text!
        k=chunk_limit,
        search_kwargs={"num_candidates": max(chunk_limit, k)},
    )

    hits: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    top_score: Optional[float] = None

    for i, (doc, score) in enumerate(results):
        real_score = float(score)
        if i == 0:
            top_score = real_score
        meta = doc.metadata or {}

        # --- NEW: drop non-matching hits early ---
        if not _passes_filters(meta):
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
            "authors": _as_author_list(meta.get("authors")),
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": meta.get("url"),
        }
        hits.append(hit)

    # ---------------------------------------------------------
# STEP BM25 — Re-rank the vector hits using BM25
# ---------------------------------------------------------
    if hits:
        bm25_scores = _bm25_rerank(hits, refined_q)
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores.values()) or 1.0


        for idx, h in enumerate(hits):
            h["bm25"] = bm25_scores[idx]

        # Combine vector score + bm25 score
        # You can tune weights later
        for h in hits:
            vec = h["score"]
            bm = h["bm25"]
            bm_norm = bm / max_bm25
            # weighted fusion
            h["rerank_score"] = 0.7 * vec + 0.3 * (bm / (bm + 1e-6))  # normalize BM25

        # Sort the hits again using rerank score
        hits.sort(key=lambda x: x["rerank_score"], reverse=True)
    
    # ---------------------------------------------------------
# STEP CROSS-ENCODER — Deep relevance re-rank
# ---------------------------------------------------------
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
            bm = h.get("bm25", 0.0)
            ce = h["cross_encoder"]

            # normalize BM25 for stability
            bm_norm = bm / (bm + 1e-6)

            h["final_score"] = (
                0.15 * vec
                + 0.1 * bm_norm
                + 0.75 * ce            # cross encoder HEAVILY dominates (expected)
            )

        hits.sort(key=lambda x: x["final_score"], reverse=True)



    # build paper view / explanations as before
    all_papers = _build_paper_view(refined_q, hits, analysis)   # pass refined_q for overlap scoring
    top_k_papers = all_papers[:k]

    for p in top_k_papers:
        for c in p["evidence"]:
            context_parts.append(c["content"])

    return {
        "query": q,
        "refined_query": refined_q,          # << expose to FE for debugging
        "analysis": analysis,                # << FE <details> shows this JSON
        "top_score": top_score if show_scores else None,
        "hits": hits,
        "papers": top_k_papers,
        "context": "\n\n---\n\n".join(context_parts),
    }

