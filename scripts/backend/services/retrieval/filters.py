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
        # split "A; B, C" robustly
        parts = [p.strip() for p in val.replace(";", ",").split(",")]
        return [p for p in parts if p]
    # fallback for odd types
    return [str(val).strip()] if str(val).strip() else []

# build ES filters
def build_filters(analysis: dict) -> list[dict]:
    """
    Build Elasticsearch DSL filters from the query-analyzer output.
    """
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

def hit_matches_filters(meta: dict, filters: List[Dict], analysis: dict) -> bool:
    """
    Lightweight Python-side filter that mirrors the index DSL filter logic.
    Used AFTER vector search to discard mismatches.
    """

    authors = analysis.get("authors") or []
    venues  = analysis.get("venues") or []
    yr      = analysis.get("time_range") or {}

    # Author match
    if authors:
        want = {a.lower() for a in authors}
        have = {a.lower() for a in as_author_list(meta.get("authors"))}
        if want.isdisjoint(have):
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