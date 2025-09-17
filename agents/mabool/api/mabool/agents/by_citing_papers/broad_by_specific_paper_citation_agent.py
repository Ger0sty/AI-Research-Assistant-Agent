from __future__ import annotations

import logging
from typing import Optional, Sequence

from ai2i.config import config_value
from ai2i.dcollection import (
    AggTransformComputedField,
    BatchComputedField,
    ComputedField,
    CorpusId,
    Document,
    DocumentCollection,
    DocumentCollectionSortDef,
    ExtractedYearlyTimeRange,
    Typed,
)
from ai2i.di import DI
from pydantic import Field

from mabool.agents.common.common import AgentState, filter_docs_by_metadata
from mabool.agents.common.sorting import weighted_sort_calculation
from mabool.agents.common.utils import alog_args
from mabool.agents.complex_search.definitions import BroadSearchInput, BroadSearchOutput
from mabool.agents.specific_paper_by_name.specific_paper_by_name_agent import (
    get_specific_paper_by_name,
)
from mabool.data_model.agent import DomainsIdentified
from mabool.data_model.config import cfg_schema
from mabool.external_api import external_api_deps
from mabool.external_api.rerank.cohere import RerankScoreDocInput, RerankScoreInput, RerankScorer
from mabool.infra.operatives import CompleteResponse, Operative, OperativeResponse
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


@DI.managed
async def sort_citing_papers_by_rerank_and_num_snippets(
    citing_papers: DocumentCollection,
    input_query: str,
    reranker: RerankScorer = DI.requires(external_api_deps.rerank_scorer),
) -> DocumentCollection:
    async def rerank_title_and_snippets(
        papers: Sequence[Document],
    ) -> list[float]:
        try:
            rerank_input = RerankScoreInput(
                query=input_query,
                docs=[
                    RerankScoreDocInput(
                        corpus_id=document.corpus_id,
                        text=(document.title if document.title else "")
                        + "\n"
                        + "\n".join([s.text for s in document.snippets if s.text]),
                    )
                    for document in papers
                    if document.snippets
                ],
            )
            rerank_results = await reranker.rerank(rerank_input=rerank_input)
            return [r.score for r in rerank_results.results]
        except Exception as e:
            logger.exception(f"Failed to rerank using cohere: {e}, defaulting to 0.0")
            return [0.0] * len(papers)

    logger.info("Reranking citing papers")
    citing_papers_with_rerank_scores = await citing_papers.with_fields(
        [
            BatchComputedField[float](
                field_name="citing_paper_rerank_score",
                required_fields=["title", "snippets"],
                computation_func=rerank_title_and_snippets,
            ),
        ],
    )
    logger.info("Adding num_snippets field")
    collection_with_num_snippets = await citing_papers_with_rerank_scores.with_fields(
        [
            ComputedField[int](
                field_name="num_snippets",
                required_fields=["snippets"],
                computation_func=Typed[Document, int](lambda doc: len(doc.snippets) if doc.snippets else 0),
            ),
        ],
    )

    async def weighted_sort_calculation_partial(
        documents: Sequence[Document],
    ) -> Sequence[float]:
        return await weighted_sort_calculation(documents, {"citing_paper_rerank_score": 1.0, "num_snippets": 0.25})

    logger.info("Adding weighted_avg_score field")
    collection_with_weighted_avg_scores = await collection_with_num_snippets.with_fields(
        [
            AggTransformComputedField[float](
                field_name="citing_paper_weighted_score",
                required_fields=["citing_paper_rerank_score", "num_snippets"],
                computation_func=weighted_sort_calculation_partial,
                cache=False,
            ),
        ],
    )
    logger.info("Sorting by citing_paper_weighted_score")
    collection_sorted = collection_with_weighted_avg_scores.sorted(
        sort_definitions=[DocumentCollectionSortDef(field_name="citing_paper_weighted_score", order="desc")]
    )
    return collection_sorted


@DI.managed
async def run_broad_by_specific_paper_citation_agent(
    content_query: str,
    domains: DomainsIdentified,
    extracted_name: str,
    anchor_doc_collection: DocumentCollection | None = None,
    time_range: Optional[ExtractedYearlyTimeRange] = None,
    venues: Optional[list[str]] = None,
    authors: Optional[list[str]] = None,
    search_iteration: int = 1,
) -> DocumentCollection:
    anchor_doc_collection = anchor_doc_collection or DC.empty()
    specific_paper_query = f"the {extracted_name} paper"
    logger.info(f"Specific paper query: {specific_paper_query}")
    specific_papers = await get_specific_paper_by_name(
        user_input=specific_paper_query,
        domains=domains,
        extracted_name=extracted_name,
        extracted_content=None,
        search_iteration=search_iteration,
        filter_threshold=config_value(cfg_schema.specific_paper_by_name_agent.filter_threshold),
    )

    if not specific_papers and not anchor_doc_collection:
        logger.info("No specific papers found, returning empty collection")
        return DC.empty()

    logger.info(
        f"Found {len(specific_papers)} specific paper candidates (taking 1st), "
        f"and got {len(anchor_doc_collection)} anchor docs"
    )
    specific_papers = specific_papers.take(1)
    specific_papers = anchor_doc_collection.merged(specific_papers)
    paper_titles = [t for t in specific_papers.project(lambda doc: doc.title) if t is not None]

    if paper_titles is None:
        return specific_papers

    try:
        specific_paper_corpus_ids = specific_papers.project(lambda doc: doc.corpus_id)
        citing_papers_with_snippets = DC.merge(
            await custom_gather(
                *[
                    fetch_citing_papers(
                        specific_paper_corpus_id=specific_paper_corpus_id,
                        content_query=content_query,
                        authors=authors,
                        time_range=time_range,
                        venues=venues,
                        search_iteration=search_iteration,
                    )
                    for specific_paper_corpus_id in specific_paper_corpus_ids
                ]
            )
        )
    except Exception:
        # Error fetching
        citing_papers_with_snippets = DC.empty()

    citing_papers_sorted = await sort_citing_papers_by_rerank_and_num_snippets(
        citing_papers_with_snippets, content_query
    )

    citing_papers_sorted = specific_papers + citing_papers_sorted

    return citing_papers_sorted


async def fetch_citing_papers(
    specific_paper_corpus_id: CorpusId,
    content_query: str,
    authors: Optional[list[str]] = None,
    time_range: Optional[ExtractedYearlyTimeRange] = None,
    venues: Optional[list[str]] = None,
    search_iteration: int = 1,
) -> DocumentCollection:
    # NOTE: SQL fallback: no explicit citation graph available.
    # We approximate "citing papers" as papers that *mention* the anchor paper
    # by title/DOI/arXiv ID in title/abstract/text, and we generate snippets locally.
    logger.info("Fetching potential 'citing' papers from SQL via mention search")

    # 1) Resolve the anchor paper (we need its title/doi/arxiv_id to build mention queries)
    anchor_dc = await DC.from_docs([])  # placeholder init
    try:
        # If your DocumentCollection supports .get(doc_id) use that instead.
        # Otherwise, requery by corpus_id to get the anchor doc’s metadata.
        # Here we assume specific_paper_corpus_id is a stable id we can search by.
        anchor = await DC.from_sql_lookup_by_id(specific_paper_corpus_id)  # <-- implement this helper in DC
        if not anchor or len(anchor) == 0:
            logger.info("Could not resolve anchor paper in SQL; returning empty collection")
            return DC.empty()
        anchor_doc = list(anchor.documents)[0]
    except Exception:
        logger.exception("Failed to resolve anchor paper; returning empty collection")
        return DC.empty()

    anchor_title = (anchor_doc.title or "").strip()
    anchor_doi = (getattr(anchor_doc, "doi", None) or "").strip()
    anchor_arxiv = (getattr(anchor_doc, "arxiv_id", None) or "").strip()

    if not any([anchor_title, anchor_doi, anchor_arxiv]):
        logger.info("Anchor paper has no title/doi/arxiv_id; cannot run mention search")
        return DC.empty()

    # 2) Build a conservative mention query for SQL (FTS or ILIKE fallback)
    # Prefer doi/arxiv (precise), then quoted title (may be long). Keep it short if needed.
    mention_terms = []
    if anchor_doi:
        mention_terms.append(anchor_doi)
    if anchor_arxiv:
        mention_terms.append(anchor_arxiv)
    if anchor_title:
        # Use a phrase if your FTS supports it; otherwise, plain keywords still help.
        mention_terms.append(f'"{anchor_title}"')

    # Combine mention terms with the user content query for better precision
    # (papers that discuss the same topic AND mention the anchor).
    # Your DC.from_sql_search should pass this straight to FTS or websearch_to_tsquery.
    combined_query = f"{content_query} " + " ".join(mention_terms)

    # 3) Retrieve candidates from SQL
    candidates = await DC.from_sql_search(
        query=combined_query,
        limit=config_value(cfg_schema.broad_search_by_keyword_agent.results_limit) * 4,  # extra headroom
        search_iteration=search_iteration,
        time_range=time_range,
        venues=venues,
        fields_of_study=None,
        fields=None,
    )

    # 4) Lightweight author/time/venue filters (reusing your existing util)
    filtered = await filter_docs_by_metadata(
        docs=candidates,
        time_range=time_range,
        venues=venues,
        authors=authors,
    )

    # 5) Generate snippets around matches so downstream reranker can use them
    def _make_snippets(doc: Document, max_snips: int = 3, window: int = 200) -> list:
        """
        Create snippet windows around the first occurrences of mention terms.
        Priority: DOI/arXiv > exact title phrase > title keywords.
        """
        body = (doc.text or "")[:200_000]  # safety cap
        haystacks = []
        if anchor_doi:
            haystacks.append(anchor_doi)
        if anchor_arxiv:
            haystacks.append(anchor_arxiv)
        if anchor_title:
            haystacks.append(anchor_title)

        snips = []
        for needle in haystacks:
            if not needle:
                continue
            idx = body.lower().find(needle.lower())
            if idx == -1:
                continue
            start = max(0, idx - window)
            end = min(len(body), idx + len(needle) + window)
            snippet_text = body[start:end].strip()
            if snippet_text:
                snips.append({"text": snippet_text})
            if len(snips) >= max_snips:
                break

        # Fallback: if nothing matched in body, try abstract/title for short snippets
        if not snips:
            for field in [doc.abstract or "", doc.title or ""]:
                if not field:
                    continue
                if anchor_doi and anchor_doi.lower() in field.lower():
                    snips.append({"text": field})
                elif anchor_arxiv and anchor_arxiv.lower() in field.lower():
                    snips.append({"text": field})
                elif anchor_title and anchor_title.lower() in field.lower():
                    snips.append({"text": field})
                if len(snips) >= max_snips:
                    break
        return snips

    # Attach snippets in-memory (your DC may provide a with_fields hook; here we mutate doc dicts for brevity)
    docs_with_snips = []
    for d in filtered.documents:
        # Ensure a 'snippets' attr/list exists
        existing_snips = getattr(d, "snippets", None) or []
        if not existing_snips:
            new_snips = _make_snippets(d)
            # If Document is pydantic/immut, convert to dict then rebuild; adapt to your DC API as needed.
            dd = d.model_dump() if hasattr(d, "model_dump") else dict(d)
            dd["snippets"] = [{"text": s["text"]} for s in new_snips]
            docs_with_snips.append(dd)
        else:
            docs_with_snips.append(d)

    citing_papers_with_snippets = await DC.from_docs(docs_with_snips)
    logger.info(
        f"Mention search yielded {len(filtered)} candidates, "
        f"{len(citing_papers_with_snippets.filter(lambda doc: len(doc.snippets or []) > 0))} with snippets"
    )
    return citing_papers_with_snippets



class BroadBySpecificPaperCitationState(AgentState):
    search_iteration: int = Field(default=1)


class BroadBySpecificPaperCitationAgent(
    Operative[BroadSearchInput, BroadSearchOutput, BroadBySpecificPaperCitationState]
):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: BroadBySpecificPaperCitationState | None, inputs: BroadSearchInput
    ) -> tuple[BroadBySpecificPaperCitationState | None, OperativeResponse[BroadSearchOutput]]:
        if not inputs.extracted_name:
            raise ValueError("extracted_name is required")
        state = state or BroadBySpecificPaperCitationState(checkpoint=DC.empty())
        docs = state.checkpoint
        if not docs:
            documents = inputs.doc_collection
        else:
            documents = docs.merged(inputs.doc_collection)

        results = await run_broad_by_specific_paper_citation_agent(
            content_query=inputs.content_query,
            domains=inputs.domains,
            extracted_name=inputs.extracted_name,
            anchor_doc_collection=inputs.anchor_doc_collection,
            time_range=inputs.time_range,
            venues=inputs.venues,
            authors=inputs.authors,
            search_iteration=state.search_iteration,
        )
        if documents:
            results = documents.merged(results)

        operative_response = CompleteResponse(
            data=BroadSearchOutput(
                doc_collection=results,
            )
        )
        return (
            BroadBySpecificPaperCitationState(checkpoint=results, search_iteration=state.search_iteration + 1),
            operative_response,
        )
