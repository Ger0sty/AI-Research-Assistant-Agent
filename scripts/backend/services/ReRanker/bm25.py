from rank_bm25 import BM25Okapi
import re

def _bm25_rerank(chunks: list[dict], query: str) -> dict[str, float]:
    """
    Return a dict mapping each chunk-id → BM25 score.
    chunk-id = index in list.
    """
    tokenized_docs = []
    for c in chunks:
        text = (c["content"] or "").lower()
        tokens = re.findall(r"[a-z0-9]+", text)
        tokenized_docs.append(tokens)
    bm25 = BM25Okapi(tokenized_docs)
    q_tokens = re.findall(r"[a-z0-9]+", query.lower())
    scores = bm25.get_scores(q_tokens)
    # Return {chunk_index: bm25_score}
    return {i: float(scores[i]) for i in range(len(chunks))}