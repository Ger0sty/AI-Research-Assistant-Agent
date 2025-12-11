# services/llm/explanation_llm.py

import re
from typing import List, Dict, Optional

from scripts.backend.llm_utils import call_llm_json_last

from scripts.backend.services.retrieval.grouping import (
    group_hits_by_paper,
    norm_query_terms,
)
from scripts.backend.services.retrieval.paper_metadata import extract_paper_meta_from_chunk
from scripts.backend.services.retrieval.signals import PaperSignals, compute_paper_signals
from scripts.backend.services.retrieval.card_builder import compose_card, compose_explanation

from dotenv import load_dotenv
import os

# Load environment settings
USE_LLM_EXPLAIN = os.getenv("USE_LLM_EXPLAIN", "0") == "1"
MAX_EXPLAIN_PAPERS = int(os.getenv("MAX_EXPLAIN_PAPERS", "3"))


# --------------------------------------------------------------
# Build per-paper objects (grouping + signals + card + explanation)
# --------------------------------------------------------------

def build_paper_view(q: str, hits: List[Dict], analysis: Dict) -> List[Dict]:
    """
    Convert raw chunk hits into a list of per-paper objects.
    """
    q_terms = norm_query_terms(q)
    by_paper = group_hits_by_paper(hits)
    papers: List[Dict] = []

    # Global relative threshold
    global_max = max(
        (h["score"] for h in hits if isinstance(h.get("score"), (int, float))),
        default=0.0,
    )
    rel_thr = 0.8 * global_max if global_max else None

    for pid, chunks in by_paper.items():
        meta = {}
        for c in chunks:
            meta.update(extract_paper_meta_from_chunk(c))

        # Derive a usable URL if missing. Many parquet rows use numeric arXiv IDs.
        url = meta.get("url")
        if not url and pid:
            pid_str = str(pid).rstrip(".0")
            # arXiv id pattern: YYYY.NNNNN (optionally with shorter suffix)
            if re.match(r"^\\d{4}\\.\\d{4,5}(v\\d+)?$", pid_str):
                url = f"https://arxiv.org/abs/{pid_str}"

        signals = compute_paper_signals(q_terms, meta, chunks, rel_threshold=rel_thr)
        card = compose_card(meta, signals, analysis, chunks)
        explanation = compose_explanation(meta, analysis, signals, chunks)

        papers.append({
            "paper_id": pid,
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "venue": meta.get("venue"),
            "year": meta.get("year"),
            "url": url,
            "card": card,
            "signals": vars(signals),
            "evidence": chunks,
            "explanation": explanation,
        })

    # Sort by composite score
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


# --------------------------------------------------------------
# LLM enhancement helpers
# --------------------------------------------------------------

def abstract_from_chunks(chunks: List[Dict], max_chars: int = 1200) -> str:
    """
    Build a synthetic abstract by preferring explicit 'abstract' sections.
    """
    if not chunks:
        return ""

    abstract_candidates = [
        c.get("content", "")
        for c in chunks
        if "abstract" in str(c.get("section") or "").lower()
    ]

    if abstract_candidates:
        text = " ".join(abstract_candidates)
    else:
        text = " ".join((c.get("content") or "") for c in chunks[:3])

    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def llm_explain_paper(
    user_query: str,
    paper_meta: dict,
    analysis: dict,
    signals: PaperSignals,
    chunks: List[Dict],
    base_explanation: str,
) -> Optional[str]:
    """
    Ask local LLM for 3–6 sentence justification.
    """
    abstract_text = abstract_from_chunks(chunks, max_chars=900)

    sorted_chunks = sorted(
        [c for c in chunks if isinstance(c.get("score"), (int, float))],
        key=lambda c: -c["score"],
    ) or chunks

    def _short(txt: str) -> str:
        txt = re.sub(r"\s+", " ", (txt or "").strip())
        return (txt[:220] + "…") if len(txt) > 240 else txt

    evidence_snips = [_short(c.get("content", "")) for c in sorted_chunks[:2]]

    payload = {
        "user_query": user_query,
        "analysis": analysis,
        "paper": {
            "title": paper_meta.get("title"),
            "authors": paper_meta.get("authors"),
            "venue": paper_meta.get("venue"),
            "year": paper_meta.get("year"),
            "url": paper_meta.get("url"),
        },
        "signals": vars(signals),
        "abstract": abstract_text,
        "evidence_snippets": evidence_snips,
        "baseline_explanation": base_explanation,
    }

    prompt = (
        "You are part of a scientific paper search engine.\n"
        "Explain in 3–6 sentences why this paper is relevant.\n"
        "Use only information from the abstract and evidence.\n"
        "Return ONLY JSON: {\"why\": \"...\"}\n\n"
        f"Input JSON:\n{payload}\n"
    )

    try:
        result = call_llm_json_last(prompt, max_new_tokens=512)
        text = None

        if isinstance(result, dict):
            text = result.get("why")
        elif isinstance(result, str):
            text = result.strip()
        else:
            s = str(result).strip()
            if len(s) > 10:
                text = s

        return text.strip() if text else None

    except Exception as e:
        print(f"[llm_explain_paper] error: {e}")
        return None


def enrich_papers_with_llm_explanations(
    user_query: str, analysis: dict, papers: List[Dict]
) -> None:
    """
    Overwrite heuristic explanations with LLM versions for top N papers.
    """
    if not USE_LLM_EXPLAIN:
        return

    for idx, p in enumerate(papers):
        if idx >= MAX_EXPLAIN_PAPERS:
            break

        try:
            signals = PaperSignals(**p["signals"])
            meta = {
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": p["authors"],
                "venue": p["venue"],
                "year": p["year"],
                "url": p["url"],
            }
            chunks = p["evidence"]

            base_expl = p.get("explanation") or compose_explanation(meta, analysis, signals, chunks)

            llm_expl = llm_explain_paper(
                user_query=user_query,
                paper_meta=meta,
                analysis=analysis,
                signals=signals,
                chunks=chunks,
                base_explanation=base_expl,
            )

            explanation = llm_expl or base_expl
            p["explanation"] = explanation

            if p.get("card"):
                p["card"]["justification"] = explanation

        except Exception as e:
            print(f"[enrich_papers] error: {e}")
            continue
