from scripts.backend.services.retrieval.filters import as_author_list

def extract_paper_meta_from_chunk(chunk: dict) -> dict:
    """
    Merge chunk-level metadata into a canonical paper-level dict.
    This normalizes metadata across chunks belonging to the same paper.
    """
    return {
        "paper_id": chunk.get("paper_id") or chunk.get("source"),
        "title": chunk.get("title"),
        "authors": as_author_list(chunk.get("authors")),
        "venue": chunk.get("venue"),
        "year": chunk.get("year"),
        "url": chunk.get("url"),
        "source": chunk.get("source"),
    }