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

# build ES filters
def build_filters(analysis: dict) -> list[dict]:
    """
    Build Elasticsearch DSL filters from the query-analyzer output.
    """
    filters: list[dict] = []
    authors = _author_terms(analysis.get("authors") or [])
    venues  = analysis.get("venues") or []
    yr      = (analysis.get("time_range") or {}) if isinstance(analysis.get("time_range"), dict) else {}

    if authors:
        filters.append({
            "bool": {
                "should": [
                    {"terms": {"metadata.authors.keyword": authors}},
                    {"terms": {"authors.keyword": authors}},
                ],
                "minimum_should_match": 1,
            }
        })
    if venues:
        normalized = [v.upper() for v in venues]
        filters.append({
            "bool": {
                "should": [
                    {"terms": {"metadata.venue.keyword": normalized}},
                    {"terms": {"venue.keyword": normalized}},
                ],
                "minimum_should_match": 1,
            }
        })
    start, end = yr.get("start"), yr.get("end")
    if start or end:
        rng = {"gte": start} if start else {}
        if end: rng["lte"] = end
        filters.append({
            "bool": {
                "should": [
                    {"range": {"metadata.year": rng}},
                    {"range": {"year": rng}},
                ],
                "minimum_should_match": 1,
            }
        })
    return filters

def hit_matches_filters(meta: dict, filters: List[Dict], analysis: dict) -> bool:
    """
    Lightweight Python-side filter that mirrors the index DSL filter logic.
    Used AFTER vector search to discard mismatches.
    """

    authors = _author_terms(analysis.get("authors") or [])
    venues  = analysis.get("venues") or []
    yr      = analysis.get("time_range") or {}

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

    return True
