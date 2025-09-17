from ai2i.dcollection import CorpusId, DenseDataset, DocumentCollection

from mabool.agents.paper_finder.definitions import PaperFinderInput
from mabool.data_model.agent import AnalyzedQuery, PartiallyAnalyzedQuery
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC


async def enrich_anchor_documents(
    analyzed_input: AnalyzedQuery | PartiallyAnalyzedQuery,
    inputs: PaperFinderInput,
) -> DocumentCollection:
    # nothing to enrich
    if not inputs.anchor_corpus_ids:
        return DC.empty()

    # fetch per-anchor; tolerate individual failures
    results = await custom_gather(
        *(
            _enrich_anchor_document(query=analyzed_input.content or inputs.query, corpus_id=corpus_id)
            for corpus_id in inputs.anchor_corpus_ids
        ),
        return_exceptions=True,
    )

    # keep only successful collections
    collections = [dc for dc in results if isinstance(dc, DocumentCollection)]
    if not collections:
        return DC.empty()

    merged = DC.merge(collections)
    return await merged.with_fields(["markdown"])


async def _enrich_anchor_document(query: str, corpus_id: CorpusId) -> DocumentCollection:
    # Restrict to the given corpus_id; vespa/open-nora/pa1-v1 is our anchor index
    return await DC.from_dense_retrieval(
        queries=[query],
        search_iteration=0,
        top_k=2,
        dataset=DenseDataset(provider="vespa", name="open-nora", variant="pa1-v1"),
        corpus_ids=[corpus_id],
    )
