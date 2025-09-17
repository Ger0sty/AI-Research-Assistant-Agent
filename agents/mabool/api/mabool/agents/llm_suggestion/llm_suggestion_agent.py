import logging
from typing import Sequence

from ai2i.chain import LLMEndpoint, LLMModel, Timeouts, define_llm_endpoint
from ai2i.config import config_value
from ai2i.dcollection import DocumentCollection, ExtractedYearlyTimeRange, OriginQuery
from ai2i.di import DI
from pydantic import BaseModel
from ai2i.config import config_value

from mabool.agents.common.common import (
    AgentState,
)
from mabool.agents.common.domain_utils import get_system_domain_params
from mabool.agents.common.utils import alog_args
from mabool.agents.llm_suggestion.llm_suggestion_prompts import (
    SuggestedPaper,
    SuggestPapersInput,
    suggested_paper,
)
from mabool.data_model.agent import (
    AgentInput,
    AgentOutput,
    DomainsIdentified,
)
from mabool.data_model.config import cfg_schema
from mabool.infra.operatives import CompleteResponse, Operative, OperativeResponse
from mabool.utils.asyncio import custom_gather
from mabool.utils.dc import DC
from mabool.utils.llm_utils import get_api_key_for_model

logger = logging.getLogger(__name__)


class LLMSuggestionArgs(BaseModel):
    user_input: str
    domains: DomainsIdentified
    extra_hints: str | None = None
    n_suggestions: int | None = 1


class LLMSuggestionInput(LLMSuggestionArgs, AgentInput):
    pass


LLMSuggestionState = AgentState
LLMSuggestionOutput = AgentOutput

# Notes
# Replace DC.from_sql_search(...) with your actual SQL retrieval helper (e.g., DC.from_sql_by_title(...)) if you have one. The signature above mirrors your earlier usage of the S2 helpers.

# Keeping the OriginQuery as "llm"/"openai" is correct — it indicates how the candidate came to be (LLM suggestion), not which backend executed the lookup.

# If you don’t want S2 fallback at all, you can drop that branch and raise if retriever.type != "sql".

def get_default_endpoint() -> LLMEndpoint:
    llm_model = LLMModel.from_name(config_value(cfg_schema.llm_suggestion_agent.llm_model_name))
    return define_llm_endpoint(
        default_timeout=Timeouts.medium,
        default_model=llm_model,
        logger=logger,
        api_key=get_api_key_for_model(llm_model),
    )


async def _fetch_by_title_backend(
    suggested_papers: Sequence[SuggestedPaper],
) -> DocumentCollection:
    """
    Fetch papers by title from the configured backend.
    - SQL path: use your SQL retriever (example shows DC.from_sql_search).
    - S2 path: fall back to DC.from_s2_by_title.
    """
    try:
        retriever_type = config_value(cfg_schema.retriever.type)  # e.g., "sql" or "s2"
    except Exception:
        retriever_type = "s2"

    if retriever_type == "sql":
        # Adjust this to your actual SQL retrieval function name/signature.
        # Using a generic keyword search against Title/Abstract/Text with a tight time window.
        search_result_futures = []
        for sp in suggested_papers:
            tr = (
                ExtractedYearlyTimeRange(start=sp.year - 2, end=sp.year + 2)
                if sp.year is not None
                else None
            )
            # If you have a title-only SQL method, prefer that (e.g., DC.from_sql_by_title).
            # Otherwise, use your general SQL search and let it match title strongly.
            search_result_futures.append(
                DC.from_sql_search(
                    query=sp.title,
                    limit=10,
                    search_iteration=1,
                    time_range=tr,
                    venues=None,
                    fields_of_study=None,  # not used in SQL
                    # optionally: title_only=True (if your helper supports it)
                )
            )
        search_results = await custom_gather(*search_result_futures)
        return DC.merge(search_results) if search_results else DC.from_docs([])

    # Fallback: original S2 behavior
    search_result_futures = [
        DC.from_s2_by_title(
            sp.title,
            time_range=(
                ExtractedYearlyTimeRange(start=sp.year - 2, end=sp.year + 2)
                if sp.year is not None
                else None
            ),
        )
        for sp in suggested_papers
    ]
    search_results = await custom_gather(*search_result_futures)
    return DC.merge(search_results) if search_results else DC.from_docs([])

# --- Keep get_default_endpoint() unchanged ---

@DI.managed
async def get_llm_suggested_papers(
    user_input: str,
    domains: DomainsIdentified,
    extra_hints: str | None = None,
    n_suggestions: int | None = None,
    search_iteration: int = 1,
) -> DocumentCollection:
    suggested_papers = (
        await get_default_endpoint()
        .execute(suggested_paper)
        .once(
            SuggestPapersInput(
                query=user_input,
                extra_hints=extra_hints or "",
                n_suggestions=n_suggestions or 1,
                **get_system_domain_params(domains),
            )
        )
    )
    if len(suggested_papers) == 0:
        return DC.empty()

    logger.info(f"Suggested papers: {suggested_papers}")

    # 🔁 Use the new backend-aware fetcher
    found_papers = await _fetch_by_title_backend(suggested_papers)

    # keep origin tagging as-is (this marks the LLM suggestion, not the retrieval backend)
    found_papers = found_papers.map_enumerate(
        lambda i, doc: doc.clone_with(
            {
                "origins": [
                    OriginQuery(
                        query_type="llm",
                        provider="openai",
                        variant=get_default_endpoint().default_model.name,
                        query=f"{user_input} | Extra hints: {extra_hints}",
                        iteration=search_iteration,
                        ranks=[i + 1],
                    )
                ]
            },
        )
    )

    return found_papers

@DI.managed
async def get_llm_suggested_papers(
    user_input: str,
    domains: DomainsIdentified,
    extra_hints: str | None = None,
    n_suggestions: int | None = None,
    search_iteration: int = 1,
) -> DocumentCollection:
    suggested_papers = (
        await get_default_endpoint()
        .execute(suggested_paper)
        .once(
            SuggestPapersInput(
                query=user_input,
                extra_hints=extra_hints or "",
                n_suggestions=n_suggestions or 1,
                **get_system_domain_params(domains),
            )
        )
    )
    if len(suggested_papers) == 0:
        return DC.empty()

    logger.info(f"Suggested papers: {suggested_papers}")
    found_papers = await _fetch_by_title_backend(suggested_papers)

    # append "llm" to the origins list
    found_papers = found_papers.map_enumerate(
        lambda i, doc: doc.clone_with(
            {
                "origins": [
                    OriginQuery(
                        query_type="llm",
                        provider="openai",
                        variant=get_default_endpoint().default_model.name,
                        query=f"{user_input} | Extra hints: {extra_hints}",
                        iteration=search_iteration,
                        ranks=[i + 1],
                    )
                ]
            },
        )
    )

    return found_papers


class LLMSuggestionAgent(Operative[LLMSuggestionInput, LLMSuggestionOutput, LLMSuggestionState]):
    def register(self) -> None: ...

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self, state: LLMSuggestionState | None, inputs: LLMSuggestionInput
    ) -> tuple[LLMSuggestionState | None, OperativeResponse[LLMSuggestionOutput]]:
        search_results = await get_llm_suggested_papers(
            inputs.user_input,
            inputs.domains,
            inputs.extra_hints,
            inputs.n_suggestions,
        )

        return (
            state,
            CompleteResponse(data=LLMSuggestionOutput(response_text="", doc_collection=search_results)),
        )
