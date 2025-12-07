from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import re
from pydantic import BaseModel, Field
from scripts.backend.llm_utils import call_llm_json

PROMPT = """
You are an expert query analyzer for a scientific paper retrieval system.

Given the user's query below, extract the following fields in JSON:
{
  "intent": "find_papers" | "find_author_papers" | "find_survey" | "other",
  "content": "main topic or keywords only",
  "authors": [list of author names],
  "venues": [list of conferences or journals],
  "years": {"start": int or null, "end": int or null},
  "recency": "recent" | "early" | null,
  "centrality": "seminal" | "less_cited" | null,
  "rationale": "brief reasoning in one sentence"
}

User query: "{query}"
"""

# Models

class YearRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    def non_empty(self) -> bool:
        return self.start is not None or self.end is not None

class ExtractedProperties(BaseModel):
    recent_first: bool = False
    recent_last: bool = False
    central_first: bool = False
    central_last: bool = False
    specific_paper_name: Optional[str] = None
    suitable_for_by_citing: Optional[bool] = None

class QueryType(BaseModel):
    type: Literal[
        "BY_AUTHOR",
        "BROAD_BY_DESCRIPTION",
        "SPECIFIC_BY_NAME",
        "SPECIFIC_BY_TITLE",
        "METADATA_ONLY_NO_AUTHOR",
        "UNKNOWN",
    ] = "UNKNOWN"
    broad_or_specific: Literal["broad", "specific", "unknown"] = "unknown"

class AnalyzerOut(BaseModel):
    original_query: str
    content: str = ""
    authors: List[str] = Field(default_factory=list)
    venues: List[str] = Field(default_factory=list)
    time_range: YearRange = Field(default_factory=YearRange)
    required_terms: List[str] = Field(default_factory=list)  # free-form must-have terms
    extracted_properties: ExtractedProperties = Field(default_factory=ExtractedProperties)
    query_type: QueryType = Field(default_factory=QueryType)
    refined_query: str = ""  # final text to search

# Hueristics
_BY_SPLIT = re.compile(r"\b(?:papers?|work|publications?)\s+by\s+", re.I)
_AND_SPLIT = re.compile(r"\s*(?:,|and|&|;)\s*", re.I)
_NAME = re.compile(r"[A-Z][a-zA-Z\-\.'`]+(?:\s+[A-Z][a-zA-Z\-\.'`]+)+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

COMMON_VENUES = {"neurips","iclr","icml","acl","emnlp","naacl","kdd","chi","cvpr","eccv","iccv","icra","corl","sigir","www"}

def _heuristics(q: str) -> Dict[str, Any]:
    s = q.strip()
    authors: List[str] = []
    # “papers by …”
    parts = _BY_SPLIT.split(s, maxsplit=1)
    if len(parts) == 2:
        rhs = parts[1]
        chunks = [p.strip() for p in _AND_SPLIT.split(rhs) if p.strip()]
        for ch in chunks:
            m = _NAME.search(ch)
            if m:
                authors.append(m.group(0))
    # “… papers”
    if not authors and s.lower().endswith((" paper"," papers"," publication"," publications")):
        core = re.sub(r"\b(papers?|publications?)\b\.?$", "", s, flags=re.I).strip()
        authors = _NAME.findall(core)

    years_found = [int(y) for y in _YEAR.findall(s)]
    # handle explicit ranges like 2019-2023
    range_match = re.findall(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b", s)
    yr = YearRange()
    if range_match:
        a, b = range_match[-1]
        yr.start, yr.end = min(int(a), int(b)), max(int(a), int(b))
    elif years_found:
        yr.start, yr.end = min(years_found), max(years_found)

    venues = [tok.upper() for tok in re.findall(r"[A-Za-z]+", s) if tok.lower() in COMMON_VENUES]

    def _dedup(seq: List[str]) -> List[str]:
        seen = set(); out = []
        for x in seq:
            if x not in seen:
                seen.add(x); out.append(x)
        return out

    return {
        "content": s,
        "authors": _dedup(authors),
        "venues": _dedup(venues),
        "time_range": yr,
    }

_SYSTEM = (
    "Extract structured filters for an academic paper search. "
    "Return ONLY valid JSON with keys: "
    "content (string), authors (string[]), venues (string[]), years (number[]|null), "
    "refined_query (string), required_terms (string[] of phrases that MUST appear in relevant papers). "
    "Do not hallucinate authors; preserve given names exactly. "
    "required_terms should include key topical phrases from the user query (e.g., 'dataset', 'text generation')."
)

def _prompt(user_q: str, hints: Dict[str, Any]) -> str:
    return (
        _SYSTEM
        + "\nUser query: " + user_q
        + "\nHints: " + str({
            "authors": hints.get("authors", []),
            "venues": hints.get("venues", []),
            "time_range": hints.get("time_range").dict() if hints.get("time_range") else {},
        })
        + "\nReturn JSON only."
    )


_STOPWORDS = {
    "paper", "papers", "publication", "publications", "work", "works",
    "about", "for", "with", "in", "on", "of", "the", "and", "or", "to",
    "by", "that", "this", "these", "those", "a", "an", "find", "looking",
    "survey", "surveys", "recent", "early", "latest", "classic", "seminal",
}


def _fallback_required_terms(text: str, limit: int = 4) -> List[str]:
    """
    Lightweight fallback extractor for required terms if the LLM does not return them.
    Produces a few keyword tokens + bigrams from the content string.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)]
    tokens = [t for t in tokens if len(t) > 3 and t not in _STOPWORDS]
    uniq: List[str] = []
    seen = set()
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    bigrams: List[str] = []
    for a, b in zip(tokens, tokens[1:]):
        phrase = f"{a} {b}"
        if phrase not in seen:
            bigrams.append(phrase)
            seen.add(phrase)
    out = (bigrams + uniq)[:limit]
    return out

def analyze_query_llm(user_q: str) -> Dict[str, Any]:
    """
    Run hybrid analysis: heuristics first, then LLM refine, then merge.
    """
    hints = _heuristics(user_q)

    # LLM refinement (sync wrapper returns dict)
    llm = call_llm_json(_prompt(user_q, hints), max_new_tokens=128)
    if not isinstance(llm, dict) or "error" in llm:
        llm = {
            "content": hints["content"],
            "authors": [],
            "venues": [],
            "years": None,
            "refined_query": user_q,
            "required_terms": [],
        }

    # Merge with bias to heuristic authors
    authors = hints["authors"] or llm.get("authors", [])
    venues = list({*(hints.get("venues") or []), *[v for v in llm.get("venues", []) if isinstance(v, str)]})

    authors = [re.sub(r"\s+", " ", a).strip(" .") for a in authors if isinstance(a, str) and a.strip()]
    venues = [v.lower() for v in venues if isinstance(v, str) and v.strip()]

    # Years → YearRange
    yr_range = hints["time_range"]
    yrs = llm.get("years")
    if isinstance(yrs, list) and all(isinstance(x, int) for x in yrs) and yrs:
        yr_range = YearRange(start=min(yrs), end=max(yrs))

    # Classify query type (Asta-like)
    qt = QueryType()
    if authors:
        qt = QueryType(type="BY_AUTHOR", broad_or_specific="broad")
    elif venues or yr_range.non_empty():
        qt = QueryType(type="METADATA_ONLY_NO_AUTHOR", broad_or_specific="broad")
    else:
        tokens = user_q.strip().split()
        qt = QueryType(
            type="BROAD_BY_DESCRIPTION" if len(tokens) <= 4 else "SPECIFIC_BY_TITLE",
            broad_or_specific="broad" if len(tokens) <= 4 else "specific",
        )

    # Required terms (LLM-provided or fallback)
    req_terms = [t for t in llm.get("required_terms", []) if isinstance(t, str) and t.strip()]
    if not req_terms:
        req_terms = _fallback_required_terms(hints.get("content", ""))

    out = AnalyzerOut(
        original_query=user_q,
        content=llm.get("content") or hints["content"],
        authors=[a for a in authors if isinstance(a, str) and a.strip()],
        venues=[v for v in venues if isinstance(v, str) and v.strip()],
        time_range=yr_range,
        required_terms=[t for t in req_terms if t],
        extracted_properties=ExtractedProperties(),
        query_type=qt,
        refined_query=(llm.get("refined_query") or user_q).strip(),
    )
    return out.dict()

def build_refined_query(analysis: Dict[str, Any]) -> str:
    """
    Build a textual query for your retriever from the analysis dict.
    Prefer `analysis['refined_query']` if available.
    """
    if isinstance(analysis.get("refined_query"), str) and analysis["refined_query"].strip():
        return analysis["refined_query"].strip()

    parts = [analysis.get("content", "")]
    parts += analysis.get("authors", []) + analysis.get("venues", [])
    parts += analysis.get("required_terms", [])
    # FIX: read YearRange from 'time_range' instead of 'years'
    yr = analysis.get("time_range") or {}
    start = yr.get("start") if isinstance(yr, dict) else None
    end = yr.get("end") if isinstance(yr, dict) else None
    if start or end:
        parts.append(f"year:[{start or '*'} TO {end or '*'}]")
    return " ".join(p for p in parts if p)
    # Required terms (LLM or fallback)
    req_terms = [t for t in llm.get("required_terms", []) if isinstance(t, str) and t.strip()]
    if not req_terms:
        req_terms = _fallback_required_terms(hints.get("content", ""))
