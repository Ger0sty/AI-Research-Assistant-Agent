import re
import math
from typing import Optional

from scripts.backend.services.retrieval.signals import PaperSignals

POSITIVE_EVAL = r"\b(experiment|evaluation|evaluat(e|ion)|benchmark|results|baseline|ablation)\b"
DATASET_HINT  = r"\b(dataset|corpus|collection|annotations?|labeled|release|we (introduce|present|release))\b"
METHOD_HINT   = r"\b(approach|method|model|framework|algorithm|pipeline)\b"

def has_keyword(text: str, pattern: str) -> bool:
    """
    Return True if the given regex pattern appears in text.
    """
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def extract_short_fact(text: str, keywords: list[str], max_len: int = 180) -> Optional[str]:
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

def grade_relevance(signals: PaperSignals, meta: dict, analysis: dict) -> tuple[str, float]:
    """
    Computes a relevance score and maps it to a label.
    Returns (label, score), where score ∈ [0,1].
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
    return label, score
