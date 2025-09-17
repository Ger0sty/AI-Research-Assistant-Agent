from __future__ import annotations

import logging
from typing import Optional

from ai2i.config import config_value, ufv
from ai2i.dcollection import (
    DocumentCollection,
    ExtractedYearlyTimeRange,
)
from ai2i.di import DI

from mabool.agents.common.domain_utils import get_fields_of_study_filter_from_domains
from mabool.agents.common.utils import alog_args
from mabool.data_model.agent import (
    AgentError,
    AgentInput,
    AgentOutput,
    DomainsIdentified,
)
from mabool.data_model.config import cfg_schema
from mabool.data_model.ufs import uf
from mabool.infra.operatives import (
    CompleteResponse,
    Operative,
    OperativeResponse,
    VoidResponse,
)
from mabool.utils.dc import DC

logger = logging.getLogger(__name__)


class MetadataOnlySearchInput(AgentInput):
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    # NOTE: if venues exist, we ignore domains, as domains are often hallucinated, and venue already selects for domain.
    domains: DomainsIdentified


async def _search_by_metadata_backend(
    *,
    query: str,
    limit: int,
    time_range: Optional[ExtractedYearlyTimeRange],
    venues: Optional[list[str]],
    fields_of_study: Optional[list[str]],
    search_iteration: int = 1,
) -> DocumentCollection:
    """
    Dispatch metadata-only search to the configured backend.
    - SQL: DC.from_sql_search(...)
    - S2:  DC.from_s2_search(...)
    """
    try:
        retriever_type = config_value(cfg_schema.retriever.type)  # expected "sql" or "s2"
    except Exception:
        retriever_type = "s2"

    if retriever_type == "sql":
        # FoS is typically not available/used on SQL; pass None.
        return await DC.from_sql_search(
            query=query,
            limit=limit,
            search_iteration=search_iteration,
            time_range=time_range,
            venues=venues,
            fields_of_study=None,
        )

    # Default / fallback: Semantic Scholar (S2)
    return await DC.from_s2_search(
        query,
        limit=limit,
        search_iteration=search_iteration,
        time_range=time_range,
        venues=venues,
        fields_of_study=fields_of_study,
    )


class MetadataOnlySearchAgent(Operative[MetadataOnlySearchInput, AgentOutput, None]):
    def register(self) -> None: ...

    @DI.managed
    async def get_papers_by_metadata(
        self,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        domains: DomainsIdentified | None = None,
    ) -> DocumentCollection:
        # Require at least a time range (non-empty) or venues.
        assert venues or (time_range and not time_range.is_empty())

        # If venues are provided, ignore domains (per original comment).
        # Otherwise, derive FoS for S2 only.
        fields_of_study = None
        if not venues and domains:
            try:
                # For SQL backends we pass None; S2 can use FoS.
                if config_value(cfg_schema.retriever.type) != "sql":
                    fields_of_study = get_fields_of_study_filter_from_domains(domains)
            except Exception:
                # If config lookup fails, be conservative and keep FoS as None.
                fields_of_study = None

        limit = config_value(cfg_schema.s2_api.total_papers_limit)

        search_results = await _search_by_metadata_backend(
            query="",  # metadata-only query
            limit=limit,
            time_range=time_range,
            venues=venues,
            fields_of_study=fields_of_study,
            search_iteration=1,
        )
        return search_results

    @alog_args(log_function=logging.info)
    async def handle_operation(
        self,
        state: None,
        inputs: MetadataOnlySearchInput,
    ) -> tuple[None, OperativeResponse[AgentOutput]]:
        response_text = ""
        try:
            results = await self.get_papers_by_metadata(inputs.time_range, inputs.venues, inputs.domains)

            if not results or len(results.documents) == 0:
                # Keep existing i18n keys to avoid breaking callers (even if backend is SQL).
                response_text = ufv(uf.response_texts.metadata_agent.could_not_find_in_s2)
                if inputs.venues and len(inputs.venues) > 0:
                    response_text += ufv(uf.response_texts.metadata_agent.try_alternative)

            elif len(results.documents) == config_value(cfg_schema.s2_api.total_papers_limit):
                response_text = ufv(
                    uf.response_texts.metadata_agent.notice_limit,
                    limit=config_value(cfg_schema.s2_api.total_papers_limit),
                )

        except Exception as e:
            logger.exception("MetadataOnlySearchAgent failed", exc_info=e)
            return None, VoidResponse(error=AgentError(type="other", message=str(e)))

        return (
            None,
            CompleteResponse(data=AgentOutput(response_text=response_text, doc_collection=results)),
        )
