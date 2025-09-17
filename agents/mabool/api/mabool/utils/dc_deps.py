from ai2i.config import ConfigValue, configurable
from ai2i.dcollection import DocumentCollection
from ai2i.di import DI, create_module

from mabool.data_model.config import cfg_schema
from mabool.dal.sql_factory import SqlDocumentCollectionFactory  # ← your DAL factory

dc_module = create_module("DocumentCollection")


@dc_module.provides(scope="singleton")
async def round_doc_collection_factory(
    db_dsn: str = DI.config(cfg_schema.sql.dsn),
    db_pool_size: int = DI.config(cfg_schema.sql.pool_size),
    db_timeout: int = DI.config(cfg_schema.sql.timeout),
    cache_ttl: int = DI.config(cfg_schema.cache.ttl),
    cache_is_enabled: bool = DI.config(cfg_schema.cache.enabled),
    force_deterministic: bool = DI.config(cfg_schema.force_deterministic),
) -> SqlDocumentCollectionFactory:
    return SqlDocumentCollectionFactory(
        dsn=db_dsn,
        pool_size=db_pool_size,
        timeout=db_timeout,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
    )


@configurable
def detached_doc_collection_factory(
    db_dsn: str = ConfigValue(cfg_schema.sql.dsn),
    db_pool_size: int = ConfigValue(cfg_schema.sql.pool_size),
    db_timeout: int = ConfigValue(cfg_schema.sql.timeout),
    cache_ttl: int = ConfigValue(cfg_schema.cache.ttl),
    cache_is_enabled: bool = ConfigValue(cfg_schema.cache.enabled),
    force_deterministic: bool = ConfigValue(cfg_schema.force_deterministic),
) -> SqlDocumentCollectionFactory:
    return SqlDocumentCollectionFactory(
        dsn=db_dsn,
        pool_size=db_pool_size,
        timeout=db_timeout,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
    )


@dc_module.provides(scope="singleton")
async def empty_doc_collection(
    dmf: SqlDocumentCollectionFactory = DI.requires(round_doc_collection_factory),
) -> DocumentCollection:
    return dmf.empty()
