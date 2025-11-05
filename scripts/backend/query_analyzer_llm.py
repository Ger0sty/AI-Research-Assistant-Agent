from typing import Dict, Any
from scripts.backend.llm_utils import call_llm_json  # your thin wrapper
from scripts.backend.query_analyzer_prompts import (
    _content_extraction_prompt_tmpl,
    _author_extraction_prompt_tmpl,
    _venue_extraction_prompt_tmpl,
    _time_range_prompt_tmpl,
)


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

def analyze_query_llm(query: str) -> Dict[str, Any]:
    """
    Run LLM-based decomposition of a query into content, authors, venues, and years.
    """
    result = {}

    # Run each extractor independently
    result["content"] = call_llm_json(f"{_content_extraction_prompt_tmpl}\n\nQuery:\n{query}").get("content")
    result["authors"] = call_llm_json(f"{_author_extraction_prompt_tmpl}\n\nQuery:\n{query}").get("authors", [])
    result["venues"] = call_llm_json(f"{_venue_extraction_prompt_tmpl}\n\nQuery:\n{query}").get("venues", [])
    result["years"] = call_llm_json(f"{_time_range_prompt_tmpl}\n\nQuery:\n{query}").get("years")

    # Now build a refined query for your retriever
    result["refined_query"] = build_refined_query(result)
    return result


def build_refined_query(analysis: Dict[str, Any]) -> str:
    parts = [analysis.get("content", "")]
    parts += analysis.get("authors", []) + analysis.get("venues", [])
    yrs = analysis.get("years") or {}
    if yrs.get("start") or yrs.get("end"):
        parts.append(f"year:[{yrs.get('start') or '*'} TO {yrs.get('end') or '*'}]")
    return " ".join([p for p in parts if p])
