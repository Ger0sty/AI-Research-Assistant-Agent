from __future__ import annotations
from typing import Any, Dict, List, Optional, Literal
import re
from pydantic import BaseModel, Field
from scripts.backend.llm_utils import call_llm_json_last
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
    refined_query: str = ""

    authors: List[str] = Field(default_factory=list)
    venues: List[str] = Field(default_factory=list)
    time_range: YearRange = Field(default_factory=YearRange)
    required_terms: List[str] = Field(default_factory=list)

    recency: Optional[str] = None
    centrality: Optional[str] = None
    broad_or_specific: Optional[str] = None
    by_title_or_name: Optional[str] = None

    extracted_properties: ExtractedProperties = Field(default_factory=ExtractedProperties)
    query_type: QueryType = Field(default_factory=QueryType)


# Heuristics
_BY_SPLIT = re.compile(r"\b(?:papers?|work|publications?)\s+by\s+", re.I)
_AND_SPLIT = re.compile(r"\s*(?:,|and|&|;)\s*", re.I)
_NAME = re.compile(r"[A-Z][a-zA-Z\-\.'`]+(?:\s+[A-Z][a-zA-Z\-\.'`]+)+")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

COMMON_VENUES = {
    "neurips",
    "iclr",
    "icml",
    "acl",
    "emnlp",
    "naacl",
    "kdd",
    "chi",
    "cvpr",
    "eccv",
    "iccv",
    "icra",
    "corl",
    "sigir",
    "www",
}


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
    if not authors and s.lower().endswith((" paper", " papers", " publication", " publications")):
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
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return {
        "content": s,
        "authors": _dedup(authors),
        "venues": _dedup(venues),
        "time_range": yr,
    }


def clean_list(lst: List[str]) -> List[str]:
    return [x.strip() for x in lst if x and x.strip()]


def analyze_query_using_llm(user_q: str) -> dict:
    """
    Use the eight Asta-style prompt templates in a single LLM call.
    """
    prompt = f"""
# Query Analysis

Below are eight extraction tasks. Perform ALL of them based on the same query.

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

    # NOTE: the prompt includes example JSON blocks. Some smaller models tend to
    # echo those before producing the real answer. Using the "last" parser
    # avoids accidentally picking an example object instead of the model's
    # final JSON.
    out = call_llm_json_last(prompt)
    if not isinstance(out, dict):
        out = {}

    content = out.get("content") or ""
    authors = out.get("authors") or []
    venues = out.get("venues") or []
    tr = out.get("time_range") or {}
    refined = out.get("refined_query") or content

    return {
        "original_query": user_q,
        "content": content,
        "authors": authors,
        "venues": venues,
        "time_range": {
            "start": tr.get("start"),
            "end": tr.get("end"),
        },
        "recency": out.get("recency"),
        "centrality": out.get("centrality"),
        "broad_or_specific": out.get("broad_or_specific"),
        "by_title_or_name": out.get("by_title_or_name"),
        "refined_query": refined.strip(),
    }


_SYSTEM = (
    "Extract structured filters for an academic paper search. "
    "Return ONLY valid JSON with keys: content (string), authors (string[]), venues (string[]), "
    "years (number[]|null), refined_query (string), required_terms (string[] of phrases that MUST appear in relevant papers). "
    "Do not hallucinate authors; preserve given names exactly. "
    "required_terms should include key topical phrases from the user query (e.g., 'dataset', 'text generation')."
)


def _fallback_required_terms(text: str, limit: int = 4) -> List[str]:
    """
    Lightweight fallback extractor for required terms if the LLM does not return them.
    Produces a few keyword tokens + bigrams from the content string.
    """
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text)]
    stopwords = {
        "paper",
        "papers",
        "publication",
        "publications",
        "work",
        "works",
        "about",
        "for",
        "with",
        "in",
        "on",
        "of",
        "the",
        "and",
        "or",
        "to",
        "by",
        "that",
        "this",
        "these",
        "those",
        "a",
        "an",
        "find",
        "looking",
        "survey",
        "surveys",
        "recent",
        "early",
        "latest",
        "classic",
        "seminal",
    }
    tokens = [t for t in tokens if len(t) > 3 and t not in stopwords]
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
    Correct Asta-style hybrid query analysis:
    1) run heuristics first (explicit signals)
    2) run Asta-style LLM (semantic interpretation)
    3) merge the two with strict rules
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

    # STEP 3 — MERGE RULES
    content = llm_content if llm_content else h_content

    cleaned_authors = clean_list(h_authors)
    authors = cleaned_authors if cleaned_authors else llm_authors

    venues = list({
        *(v.lower() for v in h_venues),
        *(v.lower() for v in llm_venues if isinstance(v, str)),
    })

    if h_time.start or h_time.end:
        time_range = h_time
    else:
        time_range = YearRange(
            start=llm_time.get("start"),
            end=llm_time.get("end"),
        )

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

    # Required terms (fallback keywords for filtering)
    req_terms = _fallback_required_terms(content)

    out = AnalyzerOut(
        original_query=user_q,
        content=content,
        authors=[a for a in authors if isinstance(a, str) and a.strip()],
        venues=[v for v in venues if isinstance(v, str) and v.strip()],
        time_range=time_range,
        required_terms=req_terms,
        recency=llm_recency,
        centrality=llm_centrality,
        broad_or_specific=llm_broad_or_specific,
        by_title_or_name=llm_by_title_or_name,
        query_type=qt,
        refined_query=refined_query,
    )

    out_dict = out.dict()
    out_dict["recency_preference"] = llm_recency is not None
    out_dict["centrality_preference"] = llm_centrality is not None
    if llm_by_title_or_name == "title":
        out_dict["exact_title"] = content

    return out_dict


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
    yr = analysis.get("time_range") or {}
    start = yr.get("start") if isinstance(yr, dict) else None
    end = yr.get("end") if isinstance(yr, dict) else None
    if start or end:
        parts.append(f"year:[{start or '*'} TO {end or '*'}]")
    return " ".join(p for p in parts if p)
