import re

from scripts.backend.services.retrieval.signals import PaperSignals
from scripts.backend.services.retrieval.explanation_heuristics import (
    extract_short_fact as _extract_short_fact,
    has_keyword as _has_keyword,
    grade_relevance as _grade_relevance,
    POSITIVE_EVAL as _POSITIVE_EVAL,
    DATASET_HINT as _DATASET_HINT,
    METHOD_HINT as _METHOD_HINT,
)

def compose_explanation(paper_meta: dict, analysis: dict, signals: PaperSignals, chunks: list[dict]) -> str:
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

def compose_card(meta: dict, signals: PaperSignals, analysis: dict, chunks: list[dict]) -> dict:
    """
    Build Asta-like justification WITHOUT pasting long chunks.
    Produces: verdict, justification, tags, facts (short), url.
    """
    # gather lightweight cues from top chunk text
    top = max(chunks, key=lambda c: float(c.get("score") or 0.0)) if chunks else {}
    text = (top.get("content") or "")[:2500]  # small budget

    # tags (boolean cues → badges)
    tags = set()

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
