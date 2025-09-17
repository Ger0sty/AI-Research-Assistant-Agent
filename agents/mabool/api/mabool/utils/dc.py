from __future__ import annotations

import asyncio
from typing import Any, Iterable, Optional, Sequence

import asyncpg
from ai2i.config import config_value
from ai2i.dcollection import (
    Document,
    DocumentCollection,
    DocumentCollectionFactory,
    DocumentFieldName,
    ExtractedYearlyTimeRange,
    PaperFinderDocument,
)
from ai2i.di import DI

from mabool.data_model.config import cfg_schema
from mabool.data_model.rounds import RoundContext
from mabool.utils import context_deps, dc_deps

_pg_pool: Optional[asyncpg.Pool] = None


async def _get_pg_pool() -> asyncpg.Pool:
    """
    Lazily create/reuse a global asyncpg pool based on cfg_schema.sql settings.
    Expects:
      cfg_schema.sql.dsn        -> e.g. "postgresql://user:pass@host:5432/db"
      cfg_schema.sql.pool_size  -> int
      (optional) cfg_schema.sql.pool_min_size -> int (defaults to 1 if missing)
    """
    global _pg_pool
    if _pg_pool is None:
        dsn = config_value(cfg_schema.sql.dsn)
        max_size = config_value(cfg_schema.sql.pool_size)
        # pool_min_size may not exist in your schema; default to 1.
        try:
            min_size = config_value(getattr(cfg_schema.sql, "pool_min_size"))
        except Exception:
            min_size = 1
        _pg_pool = await asyncpg.create_pool(dsn=dsn, min_size=min_size, max_size=max_size)
    return _pg_pool


class DC:
    # ====== Factory-backed helpers (unchanged behavior) ======
    @staticmethod
    @DI.managed
    def from_ids(
        corpus_ids: list[str],
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
    ) -> DocumentCollection:
        return dcf.from_ids(corpus_ids)

    @staticmethod
    @DI.managed
    def from_docs(
        documents: Sequence[Document],
        computed_fields: dict[DocumentFieldName, Any] | None = None,
        dcf: DocumentCollectionFactory = DI.requires(
            dc_deps.round_doc_collection_factory, default_factory=dc_deps.detached_doc_collection_factory
        ),
    ) -> DocumentCollection:
        return dcf.from_docs(documents, computed_fields)

    @staticmethod
    @DI.managed
    def empty(dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory)) -> DocumentCollection:
        return dcf.empty()

    @staticmethod
    @DI.managed
    def merge(
        collections: Iterable[DocumentCollection],
        dcf: DocumentCollectionFactory = DI.requires(dc_deps.round_doc_collection_factory),
    ) -> DocumentCollection:
        return dcf.merge(collections)

    # ====== Old S2 fetchers (explicitly unsupported under SQL) ======
    @staticmethod
    async def from_s2_by_author(*args: Any, **kwargs: Any) -> DocumentCollection:  # type: ignore[override]
        raise NotImplementedError("S2 author search is not available in the SQL backend.")

    @staticmethod
    async def from_s2_by_title(*args: Any, **kwargs: Any) -> DocumentCollection:  # type: ignore[override]
        raise NotImplementedError("S2 title search is not available in the SQL backend.")

    @staticmethod
    async def from_s2_search(*args: Any, **kwargs: Any) -> DocumentCollection:  # type: ignore[override]
        raise NotImplementedError("S2 relevance search is not available in the SQL backend.")

    @staticmethod
    async def from_s2_citing_papers(*args: Any, **kwargs: Any) -> DocumentCollection:  # type: ignore[override]
        raise NotImplementedError("S2 citing-papers fetch is not available in the SQL backend.")

    @staticmethod
    async def from_dense_retrieval(*args: Any, **kwargs: Any) -> DocumentCollection:  # type: ignore[override]
        raise NotImplementedError("Dense retrieval is not wired for the SQL backend in this module.")

    # ====== New SQL search ======
    @classmethod
    async def from_sql_search(
        cls,
        *,
        query: str,
        limit: int,
        search_iteration: int = 1,  # kept for parity, unused here
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,            # ignored: schema doesn't have venues
        fields_of_study: list[str] | None = None,   # ignored: schema doesn't have FoS
        authors: list[str] | None = None,           # optional author filter (name substrings)
        fields: list[str] | None = None,            # optional; ignored if not in schema
        request_context: RoundContext | None = DI.requires(context_deps.request_context),  # not used here
    ) -> DocumentCollection:
        """
        Run a Postgres search against a `papers` table with columns:
          - arxiv_id TEXT
          - doi TEXT
          - title TEXT
          - text TEXT
          - paper_date DATE or TIMESTAMP
          - author_names TEXT  (e.g., 'Alice Smith; Bob Lee')
          - abstract TEXT
          - (optional) doc_tsv TSVECTOR for FTS

        Config:
          - cfg_schema.sql.use_fts: bool (default True) — use doc_tsv + websearch_to_tsquery
        """
        pool = await _get_pg_pool()

        # WHERE clause assembly
        where_parts: list[str] = []
        params: list[Any] = []
        p = 1  # asyncpg 1-based parameter positions

        # Time range uses ExtractedYearlyTimeRange with .start / .end (None if open)
        if time_range:
            if getattr(time_range, "start", None) is not None:
                where_parts.append(f"EXTRACT(YEAR FROM paper_date) >= ${p}")
                params.append(time_range.start)
                p += 1
            if getattr(time_range, "end", None) is not None:
                where_parts.append(f"EXTRACT(YEAR FROM paper_date) <= ${p}")
                params.append(time_range.end)
                p += 1

        # Author filter (simple ILIKE on split author_names)
        if authors:
            ors = [f"a ILIKE ${p + i}" for i in range(len(authors))]
            where_parts.append(
                "EXISTS (SELECT 1 FROM unnest(string_to_array(coalesce(author_names,''), ';')) AS a "
                f"WHERE {' OR '.join(ors)})"
            )
            params.extend([f"%{a.strip()}%" for a in authors])
            p += len(authors)

        # Full-text vs ILIKE fallback
        use_fts = True
        try:
            use_fts = bool(config_value(getattr(cfg_schema.sql, "use_fts")))
        except Exception:
            # default to FTS if available
            use_fts = True

        rank_expr = "0.0"
        if use_fts:
            where_parts.append(f"doc_tsv @@ websearch_to_tsquery('english', ${p})")
            params.append(query)
            p += 1
            rank_expr = f"ts_rank(doc_tsv, websearch_to_tsquery('english', ${p-1}))"
        else:
            where_parts.append(f"(title ILIKE ${p} OR abstract ILIKE ${p} OR text ILIKE ${p})")
            params.append(f"%{query}%")
            p += 1

        where_sql = " AND ".join(where_parts) if where_parts else "TRUE"
        order_sql = f"{rank_expr} DESC, paper_date DESC"

        sql = f"""
            SELECT
                arxiv_id,
                doi,
                title,
                abstract,
                text,
                paper_date,
                author_names,
                {rank_expr} AS search_rank
            FROM papers
            WHERE {where_sql}
            ORDER BY {order_sql}
            LIMIT ${p}
        """
        params.append(limit)

        rows: Sequence[asyncpg.Record]
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        # Map rows to PaperFinderDocument so downstream code receives proper Document objects
        docs: list[Document] = []
        for r in rows:
            authors_list = [a.strip() for a in (r["author_names"] or "").split(";") if a.strip()]
            year = int(r["paper_date"].year) if r["paper_date"] else None
            # Choose a stable corpus_id; downstream code expects a string "corpus_id"
            corpus_id = r["doi"] or r["arxiv_id"] or f"sql:{hash((r['title'] or '')[:128])}"

            doc = PaperFinderDocument(
                corpus_id=str(corpus_id),
                title=r["title"],
                abstract=r["abstract"],
                text=r["text"],
                year=year,
                doi=r["doi"],
                arxiv_id=r["arxiv_id"],
                authors=[{"name": name} for name in authors_list] if authors_list else None,
                # You can stash rank/source as dynamic fields via `clone_with` later if you need them.
            )
            docs.append(doc)

        return cls.from_docs(docs)
