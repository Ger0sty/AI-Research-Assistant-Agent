from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import re
from pydantic import BaseModel, Field
from scripts.backend.llm_utils import call_llm_json
from scripts.backend.query_analyzer_prompts import (
    _content_extraction_prompt_tmpl,
    _author_extraction_prompt_tmpl,
    _venue_extraction_prompt_tmpl,
    _time_range_prompt_tmpl,
    _recency_extraction_prompt_tmpl,
    _centrality_extraction_prompt_tmpl,
    _broad_or_specific_query_type_prompt_tmpl,
    _by_title_or_name_query_type_prompt_tmpl,
)

# Models

class YearRange(BaseModel):
    start: Optional[int] = None
    end: Optional[int] = None
    def non_empty(self) -> bool:
        return self.start is not None or self.end is not None

class QueryType(BaseModel):
    type: Literal[
        "BY_AUTHOR",
        "BROAD_BY_DESCRIPTION",
        "SPECIFIC_BY_NAME",
        "SPECIFIC_BY_TITLE",
        "METADATA_ONLY_NO_AUTHOR",
        "UNKNOWN",
    ] = "UNKNOWN"

class AnalyzerOut(BaseModel):
    original_query: str
    content: str = ""
    refined_query: str = ""

    authors: List[str] = Field(default_factory=list)
    venues: List[str] = Field(default_factory=list)
    time_range: YearRange = Field(default_factory=YearRange)

    recency: Optional[str] = None
    centrality: Optional[str] = None
    broad_or_specific: Optional[str] = None
    by_title_or_name: Optional[str] = None

    query_type: QueryType = Field(default_factory=QueryType)
   


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

def analyze_query_using_llm(user_q: str) -> dict:
    """
    Use the Eight ( 8 ) Asta-style prompt templates, but combine them into a SINGLE
    LLM call (the way Asta actually does).
    """

    # Build a single meta-prompt
    prompt = f"""
# Query Analysis

Below are four extraction tasks. Perform ALL of them based on the same query.

## 1. CONTENT EXTRACTION
{_content_extraction_prompt_tmpl}

## 2. AUTHOR EXTRACTION
{_author_extraction_prompt_tmpl}

## 3. VENUE EXTRACTION
{_venue_extraction_prompt_tmpl}

## 4. TIME RANGE EXTRACTION
{_time_range_prompt_tmpl}

## 5. RECENCY EXTRACTION
{_recency_extraction_prompt_tmpl} 

## 6. CENTRALITY EXTRACTION
{_centrality_extraction_prompt_tmpl}

## 7. BROAD OR SPECIFIC QUERY TYPE
{_broad_or_specific_query_type_prompt_tmpl}

## 8. BY TITLE OR NAME EXTRACTION
{_by_title_or_name_query_type_prompt_tmpl}


# REQUIREMENTS
- Perform all eight (8) tasks simultaneously.
- Return a single JSON object with keys:
  content, authors, venues, time_range, refined_query, recency, centrality, broad_or_specific, by_title_or_name
- refined_query should be a clean search-friendly combination of:
  (content + authors + venues)
- Use null for missing fields.
- Do not hallucinate authors.
- Use the user's query for all sections.

# USER QUERY
{user_q}

# RETURN FORMAT
{{
  "content": "...",
  "authors": [...],
  "venues": [...],
  "time_range": {{"start": ..., "end": ...}},
  "refined_query": "...",
  "recency": "...",
  "centrality": "...",
  "broad_or_specific": "...",
  "by_title_or_name": "..."
}}
"""

    out = call_llm_json(prompt)

    # Guarantee safe dictionary
    if not isinstance(out, dict):
        out = {}

    # Normalize the model output
    content   = out.get("content") or ""
    authors   = out.get("authors") or []
    venues    = out.get("venues") or []
    tr        = out.get("time_range") or {}
    refined   = out.get("refined_query") or content

    return {
        "original_query": user_q,
        "content": content,
        "authors": authors,
        "venues": venues,

        "time_range": {
            "start": tr.get("start"),
            "end": tr.get("end"),
        },

        # NEW AST FIELDS
        "recency": out.get("recency"),
        "centrality": out.get("centrality"),
        "broad_or_specific": out.get("broad_or_specific"),
        "by_title_or_name": out.get("by_title_or_name"),

        # Always include refined query
        "refined_query": refined.strip(),
    }

_SYSTEM = (
    "Extract structured filters for an academic paper search. "
    "Return ONLY valid JSON with keys: content (string), authors (string[]), venues (string[]), "
    "years (number[]|null), refined_query (string). "
    "Do not hallucinate authors; preserve any given names exactly."
)
 

def analyze_query_llm(user_q: str) -> Dict[str, Any]:
    """
    Correct Asta-style hybrid query analysis:
    1. run heuristics first (explicit signals)
    2. run Asta-style LLM (semantic interpretation)
    3. merge the two with strict Asta rules
    """

    # STEP 1 — HEURISTICS FIRST
    hints = _heuristics(user_q)

    h_content = hints["content"]
    h_authors = hints["authors"]
    h_venues = hints["venues"]
    h_time = hints["time_range"]

    # STEP 2 — LLM SECOND
    asta = analyze_query_using_llm(user_q)

    llm_content = asta.get("content", "") or ""
    llm_authors = asta.get("authors", [])
    llm_venues = asta.get("venues", [])
    llm_time = asta.get("time_range", {})
    llm_refined = asta.get("refined_query", llm_content).strip()

    # NEW FIELDS from LLM (Asta-style)
    llm_recency = asta.get("recency")
    llm_centrality = asta.get("centrality")
    llm_broad_or_specific = asta.get("broad_or_specific")
    llm_by_title_or_name = asta.get("by_title_or_name")


    # STEP 3 — MERGE RULES (Asta-accurate)

    # CONTENT → LLM wins unless empty
    content = llm_content if llm_content else h_content

    # AUTHORS → heuristics override LLM if present
    if h_authors:
        authors = h_authors
    else:
        authors = llm_authors

    # VENUES → combine both
    venues = list({
        *(v.lower() for v in h_venues),
        *(v.lower() for v in llm_venues),
    })

    # TIME RANGE → heuristics override explicit years
    if h_time.start or h_time.end:
        time_range = h_time
    else:
        time_range = YearRange(
            start=llm_time.get("start"),
            end=llm_time.get("end")
        )

    # refined_query = always LLM cleaned form
    refined_query = llm_refined if llm_refined else content

    # Query type inference
    if authors:
        qt = QueryType(type="BY_AUTHOR", broad_or_specific="broad")
    elif venues or time_range.non_empty():
        qt = QueryType(type="METADATA_ONLY_NO_AUTHOR", broad_or_specific="broad")
    else:
        tokens = user_q.strip().split()
        qt = QueryType(
            type="BROAD_BY_DESCRIPTION" if len(tokens) <= 4 else "SPECIFIC_BY_TITLE",
            broad_or_specific="broad" if len(tokens) <= 4 else "specific",
        )

    # Build final output
    out = AnalyzerOut(
        original_query=user_q,
        content=content,
        authors=authors,
        venues=venues,
        time_range=time_range,

        # NEW FIELDS from LLM
        recency=llm_recency,
        centrality=llm_centrality,
        broad_or_specific=llm_broad_or_specific,
        by_title_or_name=llm_by_title_or_name,
        query_type=qt,
        refined_query=refined_query,
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
    # FIX: read YearRange from 'time_range' instead of 'years'
    yr = analysis.get("time_range") or {}
    start = yr.get("start") if isinstance(yr, dict) else None
    end = yr.get("end") if isinstance(yr, dict) else None
    if start or end:
        parts.append(f"year:[{start or '*'} TO {end or '*'}]")
    return " ".join(p for p in parts if p)
