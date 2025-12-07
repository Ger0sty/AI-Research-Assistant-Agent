import re
from typing import List, Dict

def as_author_list(val) -> List[str]:
    """
    Normalize authors: ensure list[str], splitting "A; B, C" formats.
    """
    if val is None:
        return []
    if isinstance(val, list):
        # ensure stringified
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        # prefer explicit separators; avoid blowing up "Last, First" into two names
        if ";" in val:
            parts = val.split(";")
        elif re.search(r"\band\b", val, re.I):
            parts = re.split(r"\band\b", val, flags=re.I)
        elif val.count(",") >= 2:
            parts = val.split(",")
        else:
            parts = [val]
        cleaned = [" ".join(p.replace(",", " ").split()) for p in parts]
        return [p for p in cleaned if p]
    # fallback for odd types
    return [str(val).strip()] if str(val).strip() else []


def _author_terms(authors: List[str]) -> List[str]:
    """
    Expand author filters to include token-level matches (e.g., "Dan Weld" -> ["Dan Weld", "Dan", "Weld"])
    so we can match both full names and split tokens stored in the index.
    """
    out: List[str] = []
    for a in authors:
        if not isinstance(a, str):
            continue
        name = a.strip()
        if not name:
            continue
        out.append(name)
        out.extend(
            [t.strip() for t in re.split(r"[\s,;]+", name) if t.strip()]
        )

    dedup: List[str] = []
    seen = set()
    for a in out:
        key = a.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(a)
    return dedup

# -----------------------------------------------
#  External API: build_filters()
# -----------------------------------------------

def build_filters(analysis: dict) -> list[dict]:
    """
    Build Elasticsearch DSL filters from the query-analyzer output.
    Includes softer semantic matching & improved required-term logic.
    """
    filters: list[dict] = []

    # Extract analyzer signals
    authors     = _author_terms(analysis.get("authors") or [])
    venues      = analysis.get("venues") or []
    yr          = analysis.get("time_range") or {}
    req_terms   = _normalize_required_terms(analysis.get("required_terms"))

    # Structured filters (unchanged logic)
    if authors:
        filters.append(_author_filter(authors))

    if venues:
        filters.append(_venue_filter(venues))

    if isinstance(yr, dict) and (yr.get("start") or yr.get("end")):
        filters.append(_year_filter(yr))

    # 💡 New robust required-terms semantic filter
    if req_terms:
        filters.append(_required_terms_filter(req_terms))

    return filters

def _normalize_required_terms(raw_terms) -> list[str]:
    """
    Cleans analyzer-required terms:
    - keeps strings only
    - strips whitespace
    - removes empty items
    """
    if not raw_terms:
        return []
    return [
        t.strip()
        for t in raw_terms
        if isinstance(t, str) and t.strip()
    ]

def _author_filter(authors: list[str]) -> dict:
    return {
        "bool": {
            "should": [
                {"terms": {"metadata.authors.keyword": authors}},
                {"terms": {"authors.keyword": authors}},
            ],
            "minimum_should_match": 1,
        }
    }

def _venue_filter(venues: list[str]) -> dict:
    normalized = [v.upper() for v in venues]
    return {
        "bool": {
            "should": [
                {"terms": {"metadata.venue.keyword": normalized}},
                {"terms": {"venue.keyword": normalized}},
            ],
            "minimum_should_match": 1,
        }
    }

def _year_filter(yr: dict) -> dict:
    rng = {}
    if yr.get("start"):
        rng["gte"] = yr["start"]
    if yr.get("end"):
        rng["lte"] = yr["end"]

    return {
        "bool": {
            "should": [
                {"range": {"metadata.year": rng}},
                {"range": {"year": rng}},
            ],
            "minimum_should_match": 1,
        }
    }

def _required_terms_filter(req_terms: list[str]) -> dict:
    """
    Soft semantic filtering:
    - Uses multi_match with fuzziness to avoid brittle exact phrase matching.
    - Requires ~60% term overlap (or 100% if only one term).
    """
    # require majority of terms (not just 1)
    minimum_should_match = (
        "60%" if len(req_terms) > 1 else "100%"
    )

    should_clauses = []
    for term in req_terms:
        # softer lexical match
        should_clauses.append({
            "multi_match": {
                "query": term,
                "fields": ["text", "content"],
                "type": "best_fields",
                "operator": "OR",
                "fuzziness": "AUTO"
            }
        })

    return {
        "bool": {
            "should": should_clauses,
            "minimum_should_match": minimum_should_match
        }
    }

def hit_matches_filters(meta: dict, filters: List[Dict], analysis: dict, content: str) -> bool:
    """
    Lightweight Python-side filter that mirrors the index DSL filter logic.
    Used AFTER vector search to discard mismatches.
    """

    authors = _author_terms(analysis.get("authors") or [])
    venues  = analysis.get("venues") or []
    yr      = analysis.get("time_range") or {}
    req_terms = [t for t in (analysis.get("required_terms") or []) if isinstance(t, str) and t.strip()]

    # Author match
    if authors:
        have_list = as_author_list(meta.get("authors"))
        have_blob = " ".join(have_list).lower()
        matched = False
        for a in authors:
            tokens = [t for t in re.split(r"\s+", a.lower()) if t]
            if tokens and all(tok in have_blob for tok in tokens):
                matched = True
                break
        if not matched:
            return False

    # Venue match
    if venues:
        venue = (meta.get("venue") or "").lower()
        want_venues = [v.lower() for v in venues]

        # Exact match OR partial match (ACL 2024 matches ACL)
        if venue not in want_venues and not any(v in venue for v in want_venues):
            return False

    # Year match
    if yr:
        y = meta.get("year")
        if isinstance(y, int):
            if yr.get("start") and y < yr["start"]:
                return False
            if yr.get("end") and y > yr["end"]:
                return False
        else:
            # If the user asked for a year range but there is no year, reject.
            if yr.get("start") or yr.get("end"):
                return False

    
    # Required terms match (must satisfy same threshold as ES filter)
    text = (content or "").lower()
    if req_terms:
        match_count = 0
        for term in req_terms:
            parts = term.lower().split()
            if parts and all(p in text for p in parts):
                match_count += 1

        # emulate ES "minimum_should_match": "60%"
        required = max(1, int(len(req_terms) * 0.6))

        if match_count < required:
            return False


    return True
