from __future__ import annotations

from typing import Mapping

from ai2i.common.utils.data_struct import SortedSet
from ai2i.dcollection import DenseDataset

from mabool.data_model.agent import DomainsIdentified


def get_system_domain_params(domains: DomainsIdentified) -> Mapping[str, str]:
    if "Computer Science" in [domains.main_field] + domains.key_secondary_fields:
        return {
            "domain_description": "linguistics, math, computer science, machine learning, or artificial intelligence (natural language processing or computer vision in particular)",
            "determiner_example": '"model" or "dataset" (for example, for "the BERT model" extract only "BERT")',
            "affiliation_example": 'for a github repo "huggingface/transformers" extract only "transformers"',
        }
    else:
        return {
            "domain_description": ", ".join([domains.main_field] + domains.key_secondary_fields),
            "determiner_example": '"region" (for example, for "the hippocampus region" extract only "hippocampus")',
            "affiliation_example": "for a medicine name Eliquis (Pfizer) extract only Eliquis",
        }


def get_dense_datasets_by_domains(domains: DomainsIdentified) -> list[DenseDataset]:
    return [DenseDataset("vespa", "open-nora", "pa1-v1")]


from ai2i.config import config_value
from mabool.data_model.config import cfg_schema
from ai2i.common.utils.data_struct import SortedSet
from ai2i.dcollection import DenseDataset
from mabool.data_model.agent import DomainsIdentified

def get_fields_of_study_filter_from_domains(domains: DomainsIdentified) -> list[str] | None:
    # If we're using SQL, there’s no FoS filter—tell callers not to apply one.
    try:
        if config_value(cfg_schema.retriever.type) == "sql":
            return None
    except Exception:
        pass  # default to old behavior if config not wired yet

    full_list = list(SortedSet(["Computer Science"] + [domains.main_field] + domains.key_secondary_fields))
    # S2-only cleanup; harmless elsewhere
    return [f for f in full_list if f and f != "Unknown"]
