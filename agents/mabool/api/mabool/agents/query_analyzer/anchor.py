import logging
from typing import TypedDict

from ai2i.chain import Timeouts, define_prompt_llm_call
from ai2i.dcollection import DocumentCollection
from pydantic import BaseModel

from mabool.agents.query_analyzer.query_analyzer import get_default_endpoint

logger = logging.getLogger(__name__)


class CombineAnchorInput(TypedDict):
    query: str
    anchors_markdown: str


class CombineAnchorOutput(BaseModel):
    combined_query: str


_combined_anchor_query_prompt_tmpl = """
Given the below query and a small set of anchor documents, produce a refined query.

Rules:
- Keep the original intent and wording dominant.
- Add only minimal clarifications derived from the anchors to disambiguate terms or add crisp context.
- Do NOT list titles or quote long passages; produce a single, concise query string.

Original Query:
{{query}}

Anchor Documents (brief excerpts):
{{anchors_markdown}}
"""

combined_anchor_query = define_prompt_llm_call(
    _combined_anchor_query_prompt_tmpl,
    format="mustache",
    input_type=CombineAnchorInput,
    output_type=CombineAnchorOutput,
)


def _mk_anchor_compendium(
    anchors: DocumentCollection,
    max_docs: int = 10,
    per_doc_chars: int = 800,
    total_chars: int = 8000,
) -> str:
    """Build a compact, token-friendly anchors string."""
    if not anchors:
        return ""

    lines: list[str] = []
    count = 0
    total = 0

    for doc in anchors.documents[:max_docs]:
        title = (doc.title or "").strip()
        body = (doc.markdown or "").strip()
        if not title and not body:
            continue

        snippet = body[:per_doc_chars]
        block = f"- {title}\n  {snippet}" if title else f"- {snippet}"
        if total + len(block) > total_chars:
            break
        lines.append(block)
        count += 1
        total += len(block)

    return "\n".join(lines)


async def combine_content_query_with_anchors(content_query: str, anchor_docs: DocumentCollection) -> str:
    # Nothing to add? Keep original.
    if not anchor_docs:
        return content_query

    try:
        # Ensure we have markdown; if caller didn’t load, fall back gracefully
        if not all(d.is_loaded("markdown") for d in anchor_docs.documents):
            try:
                anchor_docs = await anchor_docs.with_fields(["markdown", "title"])
            except Exception as e:
                logger.warning(f"Could not load anchor markdown/title; proceeding with existing fields: {e}")

        anchors_markdown = _mk_anchor_compendium(anchor_docs)
        if not anchors_markdown.strip():
            return content_query

        endpoint = get_default_endpoint().timeout(Timeouts.medium)
        out = await endpoint.execute(combined_anchor_query).once(
            {"query": content_query, "anchors_markdown": anchors_markdown}
        )
        return (out.combined_query or "").strip() or content_query

    except Exception as e:
        logger.exception(
            f"Failed to combine content query with anchor documents. Query='{content_query[:200]}...': {e}"
        )
        return content_query
