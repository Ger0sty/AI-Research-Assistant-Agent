import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, Self, Sequence, TypeVar, override

import ai2i.dcollection as dc
from ai2i.config import config_value
from ai2i.dcollection.interface.collection import BASIC_FIELDS, DocumentPredicate
from ai2i.dcollection.interface.document import Author, ExtractedYearlyTimeRange
from ai2i.di import DI
from pydantic import BaseModel, PrivateAttr, model_validator

from mabool.data_model.config import cfg_schema
from mabool.data_model.rounds import RoundContext
from mabool.utils import context_deps

from . import high

logger = logging.getLogger(__name__)

my_dir = Path(__file__).parent
runs_dir = my_dir / "experiments"
my_mod_name = __name__.split(".")[-1]
vcr_dir = runs_dir / "cassettes" / my_mod_name

# -------- Backend helpers -------- #

def _backend() -> str:
    try:
        return config_value(cfg_schema.retriever.type)  # "sql" or "s2"
    except Exception:
        return "s2"

def _ops_fields_for_backend() -> list[str]:
    """
    Fields we ask the factory to load when returning results.
    - S2 has venue, publication_venue, citation_count, fields_of_study, publication_types.
    - SQL (per your schema) has: arxiv_id, doi, title, text, paper_date, author names, abstract.
      We'll map to ai2i standard names where possible: "title", "abstract", "year" (derived from Paper Date), "authors".
    Requesting unknown fields should be avoided for SQL.
    """
    if _backend() == "s2":
        return [*BASIC_FIELDS, "publication_venue", "citation_count", "publication_types", "fields_of_study"]
    # SQL path: keep to safe basics; your ingestion should map:
    #   Paper Date -> year, Author names -> authors, Text -> markdown/fulltext if available.
    return [*BASIC_FIELDS]  # title, abstract, year, authors, etc., as available

OPS_FIELDS = _ops_fields_for_backend()

# -------- Op base types -------- #

R = TypeVar("R")

class Op(BaseModel, Generic[R]):
    _result: R | None = PrivateAttr(default=None)
    _factory: dc.DocumentCollectionFactory = PrivateAttr()

    def build(self, factory: dc.DocumentCollectionFactory) -> Self:
        self._factory = factory
        for _, v in self:
            match v:
                case Op():
                    v.build(factory)
                case Iterable():
                    for item in v:
                        if isinstance(item, Op):
                            item.build(factory)
                case _:
                    continue
        return self

    @property
    def factory(self) -> dc.DocumentCollectionFactory:
        return self._factory

    @abstractmethod
    async def run(self) -> R: ...

    async def __call__(self) -> R:
        if self._result is None:
            logger.debug(f"Started {self}")
            self._result = await self.run()
        return self._result

class DocOp(Op[dc.DocumentCollection], ABC):
    async def __call__(self) -> dc.DocumentCollection:
        result = await super().__call__()
        logger.debug(f"Returned {len(result.documents)} documents.")
        return result

@dataclass
class AuthorsCollection:
    authors: Sequence[Author]

class AuthorOp(Op[AuthorsCollection], ABC):
    async def __call__(self) -> AuthorsCollection:
        result = await super().__call__()
        logger.debug(f"Returned {len(result.authors)} authors.")
        return result

# -------- Planning scaffolding -------- #

class Plan(DocOp):
    action: str
    depends_on: Sequence[DocOp] = []

    async def run(self) -> dc.DocumentCollection:
        raise NotImplementedError(f"{self.action} is not implemented yet.")

class Union(DocOp):
    items: Sequence[DocOp]

    async def run(self) -> dc.DocumentCollection:
        colls = await asyncio.gather(*(item() for item in self.items))
        return self.factory.merge(colls)

class Intersect(DocOp):
    items: Sequence[DocOp]

    async def run(self) -> dc.DocumentCollection:
        colls = await asyncio.gather(*(item() for item in self.items))
        corpus_ids_sets = [{doc.corpus_id for doc in coll.documents} for coll in colls]
        intersected_corpus_ids = set.intersection(*corpus_ids_sets) if corpus_ids_sets else set()
        intersect = self.factory.from_ids(corpus_ids=list(intersected_corpus_ids))
        return await intersect.with_fields(OPS_FIELDS)

# -------- Backend-agnostic document fetchers -------- #

class FromByAuthorByName(DocOp):
    """
    Backend-agnostic search for papers by an author's name (not disambiguated).
    - S2: uses s2_client().search_author + from_s2_by_author (original behavior).
    - SQL: approximates via text/metadata search on authors field.
    """
    author: str

    async def run(self) -> dc.DocumentCollection:
        backend = _backend()
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)

        if backend == "s2":
            # Original S2 flow
            author_profiles = await self.factory.s2_client().search_author(query=self.author)
            by_author = await self.factory.from_s2_by_author(
                authors_profiles=[list(author_profiles)],
                limit=10_000,
                inserted_before=request_context.inserted_before if request_context else None,
            )
            return await by_author.with_fields(OPS_FIELDS)

        # SQL PATH: emulate by keyword searching author name in metadata
        # Adjust to your SQL search signature if needed (e.g., author filter param)
        results = await self.factory.from_sql_search(
            query=self.author,
            limit=10_000,
            time_range=None,
            venues=None,
            fields_of_study=None,
            search_iteration=1,
        )
        # Optional: post-filter to reduce false positives (simple case-insensitive match)
        results = results.filter(
            lambda d: any(self.author.lower() in (a.name or "").lower() for a in (d.authors or []))
        )
        return await results.with_fields(OPS_FIELDS)

class FromByAuthorById(DocOp):
    """
    Backend-agnostic search for papers by a (possibly) unique author id.
    - S2: author_id from S2 profile.
    - SQL: if you don't have stable author IDs, this will fall back to name search.
    """
    author: Author

    async def run(self) -> dc.DocumentCollection:
        backend = _backend()
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)

        if backend == "s2":
            if not self.author.author_id:
                raise ValueError("Author must have an author_id.")
            # Build a minimal S2 profile shape
            s2_profile = {"authorId": self.author.author_id, "name": self.author.name}
            by_author = await self.factory.from_s2_by_author(
                authors_profiles=[[type("S2Author", (), {"authorId": s2_profile["authorId"], "name": s2_profile["name"]})]],
                limit=10_000,
                inserted_before=request_context.inserted_before if request_context else None,
            )
            return await by_author.with_fields(OPS_FIELDS)

        # SQL PATH: no stable author_id → fallback to name match
        name = self.author.name or ""
        results = await self.factory.from_sql_search(
            query=name,
            limit=10_000,
            time_range=None,
            venues=None,
            fields_of_study=None,
            search_iteration=1,
        )
        results = results.filter(
            lambda d: any(name.lower() == (a.name or "").lower() for a in (d.authors or []))
        )
        return await results.with_fields(OPS_FIELDS)

class FromByTitle(DocOp):
    """
    Backend-agnostic search for a specific paper by title (with optional time_range/venues).
    """
    name: str
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None

    async def run(self) -> dc.DocumentCollection:
        backend = _backend()
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)

        if backend == "s2":
            candidates = await self.factory.from_s2_by_title(
                query=self.name,
                time_range=self.time_range,
                venues=self.venues,
            )
            if candidates.documents:
                return await candidates.with_fields(OPS_FIELDS)

            candidates = await self.factory.from_s2_search(
                query=self.name,
                limit=10,
                inserted_before=request_context.inserted_before if request_context else None,
            )
            results = candidates.filter(lambda doc: (doc.title or "").lower().startswith(self.name.lower() + ":"))
            return await results.with_fields(OPS_FIELDS)

        # SQL PATH
        candidates = await self.factory.from_sql_search(
            query=self.name,
            limit=50,
            time_range=self.time_range,
            venues=self.venues,
            fields_of_study=None,
            search_iteration=1,
        )
        # Prefer exact/startswith matches
        lowered = self.name.lower()
        exact = candidates.filter(lambda d: (d.title or "").lower() == lowered)
        if exact.documents:
            return await exact.with_fields(OPS_FIELDS)
        prefix = candidates.filter(lambda d: (d.title or "").lower().startswith(lowered))
        return await (prefix if prefix.documents else candidates).with_fields(OPS_FIELDS)

class FromSearch(DocOp):
    """
    Backend-agnostic metadata-only search.
    - requires at least one of (time_range, venues)
    - S2: forwards fields_of_study + min_citations
    - SQL: ignores FoS/min_citations unless you’ve mapped them
    """
    time_range: ExtractedYearlyTimeRange | None = None
    venues: list[str] | None = None
    fields_of_study: list[str] | None = None
    min_citations: int | None = None

    @model_validator(mode="after")
    def _has_any(self) -> Self:
        if not any([self.time_range, self.venues]):
            raise ValueError("At least one of 'time_range' or 'venues' must be provided.")
        return self

    async def run(self) -> dc.DocumentCollection:
        backend = _backend()
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)

        if backend == "s2":
            if not self.time_range:
                logger.warning("Searching by venue without a time range may surpass the 10k results limit.")
            return await self.factory.from_s2_search(
                query="",
                limit=10_000,
                venues=self.venues,
                time_range=self.time_range,
                fields_of_study=self.fields_of_study,
                min_citations=self.min_citations,
                fields=OPS_FIELDS,
                inserted_before=request_context.inserted_before if request_context else None,
            )

        # SQL PATH
        results = await self.factory.from_sql_search(
            query="",  # metadata-only search
            limit=10_000,
            venues=self.venues,
            time_range=self.time_range,
            fields_of_study=None,  # typically not available in SQL
            search_iteration=1,
        )
        return await results.with_fields(OPS_FIELDS)

# -------- Graph enrichers (S2 only); SQL degrades gracefully -------- #

class EnrichWithReferences(DocOp):
    source: DocOp

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        if _backend() == "s2":
            return await source.with_fields(["references"])
        # SQL path: no citation graph → return as-is
        logger.info("EnrichWithReferences skipped (no citation graph on SQL backend).")
        return source

class FilterCiting(DocOp):
    source: DocOp
    to_cite: DocOp

    async def run(self) -> dc.DocumentCollection:
        if _backend() != "s2":
            logger.info("FilterCiting skipped (no citation graph on SQL backend). Returning empty result.")
            return self.factory.empty()
        docs_that_must_cite = await self.source()
        docs_to_cite = await self.to_cite()
        corpus_ids_to_cite = {doc.corpus_id for doc in docs_to_cite.documents}
        return docs_that_must_cite.filter(
            lambda doc: any(str(ref.target_corpus_id) in corpus_ids_to_cite for ref in doc.references or [])
        )

class FilterCitedBy(DocOp):
    source: DocOp
    that_cite: DocOp

    async def run(self) -> dc.DocumentCollection:
        if _backend() != "s2":
            logger.info("FilterCitedBy skipped (no citation graph on SQL backend). Returning empty result.")
            return self.factory.empty()
        docs_that_must_be_cited = await self.source()
        docs_that_cite = await self.that_cite()
        corpus_ids_that_are_cited = {
            str(ref.target_corpus_id) for doc in docs_that_cite.documents if doc.references for ref in doc.references
        }
        return docs_that_must_be_cited.filter(lambda doc: doc.corpus_id in corpus_ids_that_are_cited)

class GetAllReferences(DocOp):
    source: DocOp

    async def run(self) -> dc.DocumentCollection:
        if _backend() != "s2":
            logger.info("GetAllReferences skipped (no citation graph on SQL backend). Returning empty result.")
            return self.factory.empty()
        source = await self.source()
        source = await source.with_fields(["references"])
        corpus_ids = {str(ref.target_corpus_id) for doc in source.documents if doc.references for ref in doc.references}
        refs = self.factory.from_ids(corpus_ids=list(corpus_ids))
        return await refs.with_fields(OPS_FIELDS)

class GetAllCiting(DocOp):
    """
    Get all documents that cite the documents in the source collection.
    Limited to 10,000 citing documents per source document (S2 only).
    SQL path: returns empty (no citation graph).
    """
    limit: ClassVar[int] = 10_000
    source: DocOp
    _semaphore: asyncio.Semaphore = PrivateAttr()

    @override
    def build(self, factory: dc.DocumentCollectionFactory) -> Self:
        max_concurrent = config_value(cfg_schema.metadata_planner_agent.ops_max_concurrency)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        return super().build(factory)

    async def task(self, doc: dc.Document) -> dc.DocumentCollection:
        request_context: RoundContext | None = DI.get_dependency(context_deps.request_context)
        async with self._semaphore:
            return await self.factory.from_s2_citing_papers(
                corpus_id=doc.corpus_id,
                total_limit=self.limit,
                inserted_before=request_context.inserted_before if request_context else None,
            )

    async def run(self) -> dc.DocumentCollection:
        if _backend() != "s2":
            logger.info("GetAllCiting skipped (no citation graph on SQL backend). Returning empty result.")
            return self.factory.empty()
        source = await self.source()
        tasks = [self.task(doc) for doc in source.documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        citing_docs = [doc for doc in results if isinstance(doc, dc.DocumentCollection)]
        return await self.factory.merge(citing_docs).with_fields(OPS_FIELDS)

# -------- Author utilities (backend-agnostic where possible) -------- #

class FindAuthorByName(AuthorOp):
    """
    - S2: use s2_client().search_author.
    - SQL: try to derive from papers (search by name, then collect authors).
    """
    author: str

    async def run(self) -> AuthorsCollection:
        backend = _backend()
        if backend == "s2":
            author_profiles = await self.factory.s2_client().search_author(query=self.author)
            if not author_profiles:
                raise ValueError(f"No authors found for name: {self.author}")
            authors = [
                Author(author_id=getattr(profile, "authorId", None), name=getattr(profile, "name", None))
                for profile in author_profiles
                if getattr(profile, "name", None)
            ]
            return AuthorsCollection(authors=authors)

        # SQL PATH: approximate by searching papers and harvesting authors
        papers = await self.factory.from_sql_search(
            query=self.author,
            limit=1000,
            time_range=None,
            venues=None,
            fields_of_study=None,
            search_iteration=1,
        )
        papers = await papers.with_fields(["authors"])
        by_id = {}
        for d in papers.documents:
            for a in (d.authors or []):
                if self.author.lower() in (a.name or "").lower():
                    key = (a.author_id or a.name or "").lower()
                    by_id[key] = a
        return AuthorsCollection(authors=list(by_id.values()))

class AuthorsOfPapers(AuthorOp):
    papers: DocOp

    async def run(self) -> AuthorsCollection:
        docs = await self.papers()
        docs_with_authors = await docs.with_fields(["authors"])
        author_by_id = {
            author.author_id: author
            for doc in docs_with_authors.documents
            if doc.authors
            for author in doc.authors
            if author.author_id
        }
        # If SQL authors lack ids, fall back to unique names
        if not author_by_id:
            by_name = {}
            for doc in docs_with_authors.documents:
                for a in (doc.authors or []):
                    if a.name:
                        by_name[a.name.lower()] = a
            return AuthorsCollection(authors=list(by_name.values()))
        return AuthorsCollection(list(author_by_id.values()))

class ByAuthorsOfPapers(DocOp):
    all_authors: bool = False
    min_authors_of_papers: int | None = None
    authors: AuthorOp

    @model_validator(mode="after")
    def no_all_authors_and_min_authors(self) -> Self:
        if self.all_authors and self.min_authors_of_papers is not None:
            raise ValueError("Cannot use both 'all_authors' and 'min_authors_of_papers'.")
        return self

    def make_filter(self, author_ids: set[str]) -> DocumentPredicate:
        def filter_min_authors(doc: dc.Document) -> bool:
            assert self.min_authors_of_papers
            if not doc.authors:
                return False
            doc_author_ids = {author.author_id for author in doc.authors if author.author_id}
            # If SQL authors have no IDs, fall back to names
            if not doc_author_ids:
                doc_author_ids = { (author.name or "").lower() for author in doc.authors if author.name }
                candidate_ids = { a.lower() for a in author_ids }
            else:
                candidate_ids = author_ids
            authors_of_papers = doc_author_ids.intersection(candidate_ids)
            return len(authors_of_papers) >= self.min_authors_of_papers
        return filter_min_authors

    async def run(self) -> dc.DocumentCollection:
        authors_collection = await self.authors()
        backend = _backend()
        # Build per-author ops
        if backend == "s2":
            ops = [FromByAuthorById(author=author) for author in authors_collection.authors]
        else:
            # SQL: reuse name-based fetcher
            ops = [FromByAuthorByName(author=author.name or "") for author in authors_collection.authors if author.name]
        op_cls = Intersect if self.all_authors else Union
        op = op_cls(items=ops).build(self.factory)
        res = await op()
        if self.min_authors_of_papers:
            # Build identifiers set (prefer ids; fallback to lowercased names)
            author_ids = {
                a.author_id for a in authors_collection.authors if a.author_id
            } or { (a.name or "").lower() for a in authors_collection.authors if a.name }
            flt = self.make_filter(author_ids=set(author_ids))
            return res.filter(flt)
        else:
            return res

# -------- Metadata filters (unchanged, with safe fallbacks) -------- #

class FilterByMinTotalAuthors(DocOp):
    source: DocOp
    min_total_authors: int

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        return source.filter(lambda doc: len(doc.authors or []) >= self.min_total_authors)

class FilterByMetadata(DocOp):
    source: DocOp
    years: list[ExtractedYearlyTimeRange] | None = None
    venue: list[str] | None = None
    venue_group: list[str] | None = None
    field_of_study: str | None = None
    publication_types: list[str] | None = None
    min_citations: int | None = None

    def is_in_year_range(self, doc_year: int, year_range: ExtractedYearlyTimeRange) -> bool:
        if year_range.start and doc_year < year_range.start:
            return False
        if year_range.end and doc_year > year_range.end:
            return False
        return True

    def validate_venue(self, doc: dc.Document) -> bool:
        if not self.venue:
            raise ValueError("No venue list to validate")
        venue_names = {venue.lower() for venue in self.venue}
        if getattr(doc, "venue", None) and (doc.venue or "").lower() in venue_names:
            return True
        if not (doc.publication_venue and doc.publication_venue.alternate_names):
            return False
        for name in doc.publication_venue.alternate_names:
            if name.lower() in venue_names:
                return True
        return False

    def validate_venue_group(self, doc: dc.Document) -> bool:
        if not self.venue_group:
            raise ValueError("No venue group list to validate")
        if not getattr(doc, "venue", None):
            return False
        return any(group.lower() in (doc.venue or "").lower() for group in self.venue_group)

    def filter(self, doc: dc.Document) -> bool:
        if self.years:
            if not doc.year:
                return False
            if not any(self.is_in_year_range(doc.year, year) for year in self.years):
                return False
        if self.venue:
            # SQL docs may not have venue/publication_venue; skip such docs
            if not hasattr(doc, "venue") and not hasattr(doc, "publication_venue"):
                return False
            if not self.validate_venue(doc):
                return False
        if self.venue_group:
            if not hasattr(doc, "venue") or not self.validate_venue_group(doc):
                return False
        if self.field_of_study:
            if not getattr(doc, "fields_of_study", None):
                return False
            if self.field_of_study.lower() not in (fos.lower() for fos in (doc.fields_of_study or [])):
                return False
        if self.publication_types:
            if not getattr(doc, "publication_types", None):
                return False
            if not any(
                pub_type.lower() in (pt.lower() for pt in self.publication_types)
                for pub_type in (doc.publication_types or [])
            ):
                return False
        if self.min_citations is not None:
            if getattr(doc, "citation_count", None) is None or (doc.citation_count or 0) < self.min_citations:
                return False
        return True

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        return source.filter(self.filter)

class FilterExclude(FilterByMetadata):
    """
    Filter documents that match the exclude criteria (negated).
    """
    author: AuthorOp | None = None
    citing: DocOp | None = None
    cited_by: DocOp | None = None

    def filter(self, doc: dc.Document) -> bool:
        return not super().filter(doc)

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        if any(getattr(self, field) is not None for field in FilterByMetadata.model_fields if field != "source"):
            source = source.filter(self.filter)
        if self.author:
            authors = await self.author()
            ids = {a.author_id for a in authors.authors if a.author_id}
            names = {(a.name or "").lower() for a in authors.authors if a.name}
            source = source.filter(
                lambda d: not any(
                    (au.author_id in ids) or ((au.name or "").lower() in names) for au in (d.authors or [])
                )
            )
        if self.citing:
            if _backend() == "s2":
                citing_docs = await self.citing()
                corpus_ids_to_exclude = {doc.corpus_id for doc in citing_docs.documents}
                source = source.filter(
                    lambda d: not any(str(ref.target_corpus_id) in corpus_ids_to_exclude for ref in d.references or [])
                )
            else:
                logger.info("FilterExclude.citing ignored on SQL backend (no citation graph).")
        if self.cited_by:
            if _backend() == "s2":
                cited_by_docs = await self.cited_by()
                corpus_ids_to_exclude = {doc.corpus_id for doc in cited_by_docs.documents}
                source = source.filter(lambda d: d.corpus_id not in corpus_ids_to_exclude)
            else:
                logger.info("FilterExclude.cited_by ignored on SQL backend (no citation graph).")
        return source

class FilterByHighlyCited(DocOp):
    source: DocOp

    async def run(self) -> dc.DocumentCollection:
        source = await self.source()
        # If SQL lacks citation_count, this will just yield empty threshold and return empty safely.
        citation_counts = [getattr(doc, "citation_count", None) or 0 for doc in source.documents]
        threshold = high.highly_cited_threshold(citation_counts)
        if threshold is None:
            return self.factory.empty()
        return source.filter(lambda doc: getattr(doc, "citation_count", None) is not None and doc.citation_count >= threshold)
