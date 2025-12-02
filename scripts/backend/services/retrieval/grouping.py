from collections import defaultdict
import re

def group_hits_by_paper(hits: list[dict]) -> dict[str, list[dict]]:
    """
    Group retrieved chunk hits by paper_id.
    If a chunk has no paper_id, create a synthetic one using its source/title/row.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for h in hits:
        pid = h.get("paper_id")
        if not pid:
            src = (h.get("source") or "unknown").split("/")[-1]
            pid = f"{src}::{h.get('title') or 'untitled'}::{h.get('row') or id(h)}"
        grouped[pid].append(h)
    return grouped

def norm_query_terms(q: str) -> set[str]:
    """
    Normalize a query string into a set of meaningful lowercase tokens,
    excluding stopwords and tokens shorter than 3 characters.
    """
    toks = re.findall(r"[a-z0-9]+", q.lower())
    stop = {"the","a","an","and","or","for","of","to","in","on","with","by","from"}
    return {t for t in toks if t not in stop and len(t) >= 3}

def _name_tokens(n: str) -> list[str]:
    """
    Extract lowercase alphabetic tokens from a full name.
    """
    return re.findall(r"[a-z]+", (n or "").lower())

def surname_set(names: list[str]) -> set[str]:
    """
    Convert a list of authors (possibly multi-word names)
    into a set of lowercase surnames.
    """
    out = set()
    for n in names or []:
        toks = _name_tokens(n)
        if toks:
            out.add(toks[-1])
    return out