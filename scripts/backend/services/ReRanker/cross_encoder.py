from sentence_transformers import CrossEncoder

cross_encoder = None

def _load_cross_encoder():
    global cross_encoder
    if cross_encoder is None:
        cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return cross_encoder


def _cross_encoder_rerank(query: str, hits: list[dict]) -> dict[int, float]:
    """
    Returns: {chunk_index: cross_encoder_score}
    """
    if not hits:
        return {}
    
    ce = _load_cross_encoder()

    # Build sentence pairs: (query, chunk_content)
    pairs = [(query, h["content"]) for h in hits]

    # CrossEncoder returns relevance scores
    ce_scores = ce.predict(pairs, batch_size=8, show_progress_bar=False)


    return {i: float(ce_scores[i]) for i in range(len(hits))}