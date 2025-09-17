from __future__ import annotations

import json
import logging
import math
from functools import partial
from typing import Literal, TypedDict

from ai2i.dcollection import Document, DocumentCollection, DocumentFieldName, ExtractedYearlyTimeRange
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


ControlType = Literal["failure", "control", "result"]


class AgentState(BaseModel):
    checkpoint: DocumentCollection | None = None

    @field_validator("checkpoint", mode="before")
    @classmethod
    def copy_doc_collection(cls, doc_collection: DocumentCollection | None) -> DocumentCollection | None:
        if doc_collection:
            copied_documents = [d.model_copy(deep=True) for d in doc_collection.documents]
            return doc_collection.factory.from_docs(copied_documents)
        return doc_collection


def filter_by_time_range_with_buffer(
    docs: DocumentCollection,
    time_range: ExtractedYearlyTimeRange,
    keep_missing: bool = True,
    use_buffer: bool = False,
) -> DocumentCollection:
    if not time_range.start and not time_range.end:
        return docs
    buffer = 0
    if time_range.start and time_range.end and use_buffer:
        buffer = math.ceil((time_range.end - time_range.start + 1) * (20 / 100))  # +/-20%

    if buffer:
        logger.info(f"Adding buffer of +/- {buffer} years to the time range")

    def _is_doc_in_timerange_with_buffer(doc: Document) -> bool:
        if not doc.year:
            if keep_missing:
                logger.warning("filter_by_time_range_with_buffer: Keeping the doc even though year is not set")
                return True
            return False

        start = time_range.start - buffer if time_range.start else 0
        end = time_range.end + buffer if time_range.end else math.inf

        if start <= doc.year <= end:
            return True

        return False

    return docs.filter(_is_doc_in_timerange_with_buffer)


def filter_by_venues(doc: Document, venues: list[str], keep_missing: bool = False) -> bool:
    # Accept both S2-style structured venue and simple strings.
    found_names_lowered_set = set()

    # S2-style: doc.venue may be a string already; keep it.
    if getattr(doc, "venue", None):
        v = doc.venue
        found_names_lowered_set.add(v.lower() if isinstance(v, str) else str(v).lower())

    # S2-style structured publication_venue
    pv = getattr(doc, "publication_venue", None)
    if pv:
        # pv could be a dict-like or object; be defensive
        norm = getattr(pv, "normalized_name", None) or (pv.get("normalized_name") if isinstance(pv, dict) else None)
        if norm:
            found_names_lowered_set.add(str(norm).lower())
        alts = getattr(pv, "alternate_names", None) or (pv.get("alternate_names") if isinstance(pv, dict) else None)
        if alts:
            found_names_lowered_set.update([str(a).lower() for a in alts])

    # SQL path: if neither venue nor publication_venue present
    if not found_names_lowered_set:
        if keep_missing:
            logger.warning("filter_by_venues: Keeping doc; no venue/publication_venue present")
            return True
        return False

    requested_venues_lowered_set = {v.lower() for v in venues}
    return bool(found_names_lowered_set.intersection(requested_venues_lowered_set))



def filter_by_author(expected_authors: list[str], doc: Document, keep_missing: bool | None = False) -> bool:
    if not expected_authors:
        return True

    found_authors = getattr(doc, "authors", None)

    # SQL path: authors might be a single string like "Alice Smith; Bob Lee"
    if isinstance(found_authors, str):
        found_authors_list = [a.strip() for a in found_authors.split(";") if a.strip()]
    elif isinstance(found_authors, (list, tuple)):
        # Might be list[str] or list[objects with .name]
        tmp = []
        for a in found_authors:
            if isinstance(a, str):
                tmp.append(a.strip())
            else:
                name = getattr(a, "name", None) or (a.get("name") if isinstance(a, dict) else None)
                if name:
                    tmp.append(str(name).strip())
        found_authors_list = tmp
    else:
        found_authors_list = []

    if not found_authors_list:
        if keep_missing:
            logger.warning("filter_by_author: Keeping the doc even though authors are not set")
            return True
        return False

    # Matching: last name must match; if initials available, match them too.
    def initials(parts: list[str]) -> tuple[str, str] | tuple[()]:
        return (parts[0][0], parts[-1][0]) if len(parts) > 1 and parts[0] and parts[-1] else ()

    for expected_author in expected_authors:
        exp_parts = expected_author.lower().split()
        exp_last = exp_parts[-1] if exp_parts else ""
        exp_inits = initials(exp_parts)
        matched = False
        for fa in found_authors_list:
            f_parts = fa.lower().split()
            f_last = f_parts[-1] if f_parts else ""
            f_inits = initials(f_parts)
            if exp_last == f_last and (not exp_inits or not f_inits or exp_inits == f_inits):
                matched = True
                break
        if not matched:
            return False
    return True


async def filter_docs_by_metadata(
    docs: DocumentCollection,
    time_range: ExtractedYearlyTimeRange | None = None,
    venues: list[str] | None = None,
    authors: list[str] | None = None,
    keep_missing: bool = False,
    use_time_buffer: bool = False,
) -> DocumentCollection:
    fields_to_load: list[DocumentFieldName] = []
    if time_range and time_range.non_empty():
        fields_to_load.append("year")
    if venues:
        # These may not exist in SQL; with_fields should tolerate missing, but keep it minimal.
        fields_to_load.extend([f for f in ["venue", "publication_venue"] if hasattr(docs.factory.document_type, f)])
    if authors:
        fields_to_load.append("authors")

    if not fields_to_load:
        return docs
    else:
        logger.info(f"Number of documents before filter: {len(docs)}")
        docs = await docs.with_fields(fields_to_load)

    if time_range and time_range.non_empty():
        logger.info(f"Filtering documents by time range: {time_range.start} <= year <= {time_range.end}")
        docs = filter_by_time_range_with_buffer(docs, time_range, keep_missing, use_time_buffer)

    if venues:
        logger.info(f"Filtering documents by venue list: {', '.join(venues)}")
        docs = docs.filter(partial(filter_by_venues, venues=venues, keep_missing=keep_missing))

    if authors:
        logger.info(f"Filtering documents by author list: {', '.join(authors)}")
        docs = docs.filter(partial(filter_by_author, expected_authors=authors, keep_missing=keep_missing))

    logger.info(f"Number of documents after filter: {len(docs)}")
    return docs


class InputQuery(TypedDict):
    query: str


class InputQueryJson(InputQuery):
    query_json: str


def as_input_query(query: str) -> InputQuery:
    return {"query": query}


def as_input_query_json(query: str) -> InputQueryJson:
    return {"query_json": json.dumps({"query": query}), "query": query}
