import logging
from datetime import datetime
from typing import Any, Coroutine, Sequence

from ai2i.chain import LLMEndpoint, LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import config_value
from ai2i.dcollection import (
    DocumentCollection,
    DocumentCollectionFactory,
    DocumentCollectionSortDef,
    ExtractedYearlyTimeRange,
)
from ai2i.di import DI
from pydantic import BaseModel

from mabool.agents.broad_search_by_keyword.broad_search_by_keyword_agent import suggest_retrieval_query
from mabool.agents.common.common import AgentState, filter_by_author, filter_docs_by_metadata
from mabool.agents.common.computed_fields.fields import rerank_score_field
from mabool.agents.common.computed_fields.relevance import relevance_judgement_field
from mabool.agents.common.relevance_judgement_utils import get_relevant_docs
from mabool.agents.common.utils import alog_args
from mabool.agents.llm_suggestion.llm_suggestion_agent import get_llm_suggested_papers
from mabool.agents.search_by_authors.search_by_authors_prompts import disambiguate_user_response
from mabool.data_model.agent import (
    AgentError,
    AgentInput,
    AgentOutput,
    BroadOrSpecificLiterals,
    DomainsIdentified,
    RelevanceCriteria,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import CompleteResponse, InquiryQuestion, Operative, OperativeResponse, VoidResponse
from mabool.utils import dc_deps
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)

# ---------- Adapters you must implement against your SQL DB ----------

class AuthorProfile(BaseModel):
    author_id: str
    name: str
    paper_count: int = 0  # computed as COUNT(*) over your papers table

async def sql_find_authors_by_name(name: str, factory: DocumentCollectionFactory) -> list[AuthorProfile]:
    """
    Implement: fuzzy/exact search over your author index/table.
    SHOULD populate paper_count (e.g., count of papers for ranking).
    """
    raise NotImplementedError

# If you already have helpers, replace these two calls accordingly:
# - await factory.from_sql_authors([[AuthorProfile(...), ...]], limit=...)
# - await factory.from_sql_query(query=..., author_names=[...], start_year=..., end_year=..., limit=...)

# ---------------------------------------------------------------------

class SearchByAuthorsInput(AgentInput):
    authors: list[str]
    broad_or_specific: BroadOrSpecificLiterals
    user_content_input: str | None = None
    relevance_criteria: RelevanceCriteria | None = None
    time_range: ExtractedYearlyTimeRange | None = None
    # Kept for API compatibility, but ignored by SQL path unless you add venues to your schema.
    venues: list[str] | None = None
    # Kept for API compatibility; unused in SQL path unless you map it yourself.
    domains: DomainsIdentified


class NoAuthorMatchedError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.message = msg


SearchByAuthorsState = AgentState
SearchByAuthorsOutput = AgentOutput

AUTHOR_PAGINATION = 5
prefix = "Please specify which of the following you are interested in (out of {amount}):\n\n"
author_candidate = '{idx}. {name} — {count} papers'  # simplified; no h-index available in SQL by default
suffix = "\n\nPlease type the index of the author (can be an index also from previous responds)"
next_suffix = ', or "next" otherwise.'
no_next_suffix = " as this is the last batch of authors."


def get_default_endpoint() -> LLMEndpoint:
    llm_model = LLMModel.from_name(config_value(cfg_schema.search_by_author_agent.llm_model_name))
    return define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )


def get_time_range_hint(time_range: ExtractedYearlyTimeRange | None) -> str:
    if not time_range:
        return ""
    if time_range.start:
        if time_range.start == time_range.end:
            return f"During {time_range.start}."
        return f"Sometime between {time_range.start} and {time_range.end or datetime.now().year}."
    if time_range.end:
        return f"Sometime no later than {time_range.end}."
    return ""


def get_venues_hint(venues: list[str] | None) -> str:
    # Your SQL schema does not include venues; leave blank or wire up once available.
    return ""


class SearchByAuthorsAgent(Operative[SearchByAuthorsInput, SearchByAuthorsOutput, SearchByAuthorsState]):
    def register(self) -> None:
        ...

    async def disambiguate_author(self, author_name: str, found_authors: list[AuthorProfile]) -> list[AuthorProfile]:
        # Rank by: exact case-insensitive match > all tokens contained > paper_count
        tokens = author_name.lower().split()
        sorted_authors = sorted(
            found_authors,
            key=lambda a: (
                a.name.lower() == author_name.lower(),
                all(t in a.name.lower().split() for t in tokens),
                int(a.paper_count),
            ),
            reverse=True,
        )
        inquiry = self.inquiry()
        if (
            not config_value(cfg_schema.search_by_author_agent.disambiguate_authors)
            or inquiry is None
            or len(sorted_authors) == 1
        ):
            return sorted_authors[: config_value(cfg_schema.search_by_author_agent.consider_profiles_per_author)]

        for i in range(0, len(sorted_authors), AUTHOR_PAGINATION):
            not_last_batch = i + AUTHOR_PAGINATION < len(sorted_authors)
            cur_authors = sorted_authors[i : i + AUTHOR_PAGINATION]
            options = [str(i + j + 1) for j in range(len(cur_authors))] + (["next"] if not_last_batch else [])
            formatted_candidates = [
                author_candidate.format(idx=i + j + 1, name=a.name, count=a.paper_count) for j, a in enumerate(cur_authors)
            ]
            formulated_question = (
                prefix.format(amount=len(sorted_authors))
                + "\n".join(formatted_candidates)
                + suffix
                + (next_suffix if not_last_batch else no_next_suffix)
            )

            inquire_response = await inquiry.ask(InquiryQuestion(question=formulated_question, options=options))
            try:
                if inquire_response.answer == "next":
                    continue
                disambiguated_user_response = int(inquire_response.answer)
                return [sorted_authors[disambiguated_user_response - 1]]
            except ValueError:
                options_without_next = options[:-1] if "next" == options[-1] else options
                disambiguated_user_response = (
                    await get_default_endpoint()
                    .execute(disambiguate_user_response)
                    .once(
                        {
                            "agents_question": formulated_question,
                            "options": options_without_next,
                            "user_response": inquire_response.answer,
                        }
                    )
                )
                return [sorted_authors[disambiguated_user_response - 1]]

        raise NoAuthorMatchedError("There are no more authors matching this name")

    @DI.managed
    async def get_authors_papers_by_sql_authors(
        self,
        authors: list[str],
        doc_collection_factory: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
    ) -> DocumentCollection:
        # Find top profiles per requested author
        top_authors: list[list[AuthorProfile]] = []
        for author in authors:
            found = await sql_find_authors_by_name(author, doc_collection_factory)
            if not found:
                raise NoAuthorMatchedError(f"No author was found by this name: {author}")
            top_authors.append(await self.disambiguate_author(author, found))

        flat = [a for group in top_authors for a in group]

        # Replace with your actual factory/adapter call:
        # Expectation: returns union of papers for the provided author IDs.
        results = await doc_collection_factory.from_sql_authors(
            author_ids=[a.author_id for a in flat],
            limit=config_value(cfg_schema.s2_api.total_papers_limit),  # reuse limit knob
        )

        if len(authors) > 1:
            # Keep only papers that include at least one profile per requested author group
            results = results.filter(
                lambda doc: all(
                    len({au.author_id for au in (doc.authors or [])}.intersection({p.author_id for p in group})) > 0
                    for group in top_authors
                )
            )

        return results

    async def get_authors_papers_by_sql_relevance(
        self,
        authors: list[str],
        user_content_input: str,
        domains: DomainsIdentified,  # kept for signature compatibility
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,  # ignored; schema has no venues
    ) -> DocumentCollection:
        reformulated_query = await suggest_retrieval_query(user_content_input)
        # Replace with your SQL-backed text search over Title/Abstract/Text
        return await DC.from_sql_query(  # <-- swap to your helper
            query=reformulated_query,
            author_names=authors,
            start_year=time_range.start if time_range else None,
            end_year=time_range.end if time_range else None,
            limit=config_value(cfg_schema.search_by_author_agent.relevance_judgements_quota),
        )

    def get_authors_papers_fast_and_naive_methods(
        self,
        authors: list[str],
        user_content_input: str,
        domains: DomainsIdentified,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
    ) -> list[Coroutine[Any, Any, DocumentCollection]]:
        futures: list[Coroutine[Any, Any, DocumentCollection]] = []
        # 1) SQL full-text relevance restricted to author(s)
        futures.append(self.get_authors_papers_by_sql_relevance(authors, user_content_input, domains, time_range, venues))
        # 2) LLM suggestions + title search (still useful)
        extra_hints = [
            f"The paper was written by {' and '.join(authors)}.",
            get_time_range_hint(time_range),
            get_venues_hint(venues),
        ]
        futures.append(
            get_llm_suggested_papers(
                user_input=user_content_input,
                domains=domains,
                extra_hints=" ".join(filter(None, extra_hints)),
            )
        )
        return futures

    @DI.managed
    async def relevance(
        self,
        results: DocumentCollection,
        user_content_input: str | None,
        relevance_criteria: RelevanceCriteria | None,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,  # ignored unless you add venues to SQL
    ) -> DocumentCollection:
        if (time_range and time_range.non_empty()) or (
            user_content_input and relevance_criteria and relevance_criteria.required_relevance_critieria
        ):
            results = await filter_docs_by_metadata(results, time_range, None, keep_missing=False)

            if user_content_input and relevance_criteria and relevance_criteria.required_relevance_critieria:
                quota = config_value(cfg_schema.search_by_author_agent.relevance_judgements_quota)
                if len(results.documents) > quota:
                    results = await results.with_fields([rerank_score_field(relevance_criteria)])
                    results = results.sorted([DocumentCollectionSortDef(field_name="rerank_score", order="desc")]).take(
                        quota
                    )
                results = await results.with_fields([relevance_judgement_field(relevance_criteria)])
                results = get_relevant_docs(results)
        return results

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self,
        state: SearchByAuthorsState | None,
        inputs: SearchByAuthorsInput,
    ) -> tuple[SearchByAuthorsState | None, OperativeResponse[SearchByAuthorsOutput]]:
        try:
            futures: list[Coroutine[Any, Any, DocumentCollection]] = []

            if inputs.user_content_input:
                futures.extend(
                    self.get_authors_papers_fast_and_naive_methods(
                        inputs.authors, inputs.user_content_input, inputs.domains, inputs.time_range, inputs.venues
                    )
                )

            futures.append(self.get_authors_papers_by_sql_authors(inputs.authors))

            result_sets = await custom_gather(*futures, return_exceptions=True)
            ok_sets: list[DocumentCollection] = []
            unknown_exceptions: list[BaseException] = []

            for rs in result_sets:
                if isinstance(rs, NoAuthorMatchedError):
                    return (
                        state,
                        CompleteResponse(data=SearchByAuthorsOutput(response_text=rs.message, doc_collection=DC.empty())),
                    )
                if isinstance(rs, BaseException):
                    unknown_exceptions.append(rs)
                else:
                    ok_sets.append(rs)

            if not ok_sets:
                raise unknown_exceptions[0]

            logger.info("All results gathered, merging...")
            results: DocumentCollection = ok_sets[0]
            for s in ok_sets[1:]:
                results += s

            results = await self.relevance(
                results,
                inputs.user_content_input,
                inputs.relevance_criteria,
                inputs.time_range,
                inputs.venues,
            )

            if inputs.broad_or_specific == "specific":
                results = results.take(config_value(cfg_schema.search_by_author_agent.limit_for_specific))

        except Exception as e:
            return None, VoidResponse(error=AgentError(type="other", message=str(e)))

        return state, CompleteResponse(data=SearchByAuthorsOutput(response_text="", doc_collection=results))
