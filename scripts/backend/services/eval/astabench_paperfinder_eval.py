"""
Evaluate the retrieval stack on AstaBench's PaperFindingBench queries.

This version runs entirely on your local index and corpus:
- Uses the benchmark query text as-is (no LLM calls).
- Maps SemanticScholar CorpusIDs to your local paper IDs via arXiv IDs or titles.
- Computes Recall@k, Hits@1, and MRR on the subset of queries where at least one
  gold paper could be mapped into your corpus.

If you want fuller coverage, supply a metadata JSON mapping from CorpusID ->
{title, externalIds:{ArXiv, DOI}} (e.g., exported from Semantic Scholar).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

from scripts.backend.services.retrieval.elastic_client import wait_for_index_ready
from scripts.backend.services.pipeline.rag_pipeline import query_rag


# --------------------------
# Data helpers
# --------------------------


def _read_local_corpus(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Could not find a dataset at {parquet_path} or {csv_path}"
        )
    return df


def _norm_title(title: str) -> str:
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def _build_local_maps(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
        title_to_pid: normalized title -> paper_id
        arxiv_to_pid: arxiv_id -> paper_id
    """
    title_to_pid: Dict[str, str] = {}
    arxiv_to_pid: Dict[str, str] = {}
    for row in df.itertuples():
        pid = getattr(row, "paper_id", None) or getattr(row, "id", None) or None
        if pid is None:
            continue
        pid = str(pid)
        title = getattr(row, "title", None)
        if title:
            title_to_pid.setdefault(_norm_title(title), pid)

        # arxiv style ids may live in several columns
        for field in ["arxiv_id", "ArXiv", "arxivId", "oai_identifier", "OAI_identifier"]:
            if hasattr(row, field):
                raw = getattr(row, field)
            else:
                raw = None
            if raw and isinstance(raw, str):
                clean = raw.replace("arxiv:", "").replace("arXiv:", "")
                arxiv_to_pid.setdefault(clean, pid)
    return title_to_pid, arxiv_to_pid


def _load_meta_map(meta_path: Optional[Path]) -> Dict[str, dict]:
    """
    Load a mapping of CorpusID -> metadata dict.
    Supports either:
      - JSON object mapping ids to dicts
      - JSONL with one object per line having a 'corpusId' field
    """
    if meta_path is None:
        return {}
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    txt = meta_path.read_text(encoding="utf-8")
    txt_strip = txt.lstrip()
    if txt_strip.startswith("{"):
        data = json.loads(txt)
        return {str(k): v for k, v in data.items()}

    meta: Dict[str, dict] = {}
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        cid = str(obj.get("corpusId") or obj.get("paperId") or obj.get("id"))
        if cid:
            meta[cid] = obj
    return meta


def _map_corpus_ids_to_local(
    corpus_ids: Iterable[str],
    meta_map: Dict[str, dict],
    title_to_pid: Dict[str, str],
    arxiv_to_pid: Dict[str, str],
) -> Set[str]:
    """
    Convert SemanticScholar CorpusIDs to local paper_ids using metadata hints.
    """
    out: Set[str] = set()
    for cid in corpus_ids:
        if cid is None:
            continue
        cid = str(cid)
        meta = meta_map.get(cid, {})
        ext = meta.get("externalIds") or {}

        # Try arXiv first
        for key in ["ArXiv", "arxiv", "ARXIV", "S2ORC"]:
            ax = ext.get(key) if isinstance(ext, dict) else None
            if ax and ax in arxiv_to_pid:
                out.add(arxiv_to_pid[ax])
                break

        # Then title match
        if meta.get("title"):
            nt = _norm_title(meta["title"])
            pid = title_to_pid.get(nt)
            if pid:
                out.add(pid)
    return out


# --------------------------
# Evaluation
# --------------------------

@dataclass
class EvalResult:
    query_id: str
    query: str
    gold_local_ids: Set[str]
    found_rank: Optional[int]
    top_hits: List[dict]


def _hit_rank(hits: List[dict], gold_ids: Set[str], k: int) -> Optional[int]:
    for idx, h in enumerate(hits[:k]):
        pid = str(h.get("paper_id") or "")
        if pid in gold_ids:
            return idx + 1
    return None


def run_eval(
    bench_path: Path,
    parquet_path: Path,
    csv_path: Path,
    meta_path: Optional[Path],
    sample: Optional[int],
    seed: int,
    k: int,
) -> Tuple[dict, List[EvalResult]]:
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    if sample:
        bench = random.Random(seed).sample(bench, k=min(sample, len(bench)))

    df = _read_local_corpus(parquet_path, csv_path)
    title_to_pid, arxiv_to_pid = _build_local_maps(df)
    meta_map = _load_meta_map(meta_path)

    eval_results: List[EvalResult] = []
    total_with_gold = 0
    hits_count = 0
    hits1 = 0
    rr_sum = 0.0

    for entry in tqdm(bench, desc="Evaluating", unit="q"):
        query = entry.get("input", {}).get("query", "")
        qid = entry.get("input", {}).get("query_id", "")
        golds = entry.get("scorer_criteria", {}).get("known_to_be_good", [])
        gold_local = _map_corpus_ids_to_local(golds, meta_map, title_to_pid, arxiv_to_pid)
        if not gold_local:
            eval_results.append(EvalResult(qid, query, set(), None, []))
            continue

        total_with_gold += 1

        rag = query_rag(q=query, k=k, show_scores=False)
        top_papers = rag.get("papers") or []
        rank = _hit_rank(top_papers, gold_local, k)
        if rank is not None:
            hits_count += 1
            rr_sum += 1.0 / rank
            if rank == 1:
                hits1 += 1

        eval_results.append(EvalResult(qid, query, gold_local, rank, top_papers[:k]))

    if total_with_gold == 0:
        return {
            "error": "No AstaBench gold papers could be mapped into the local corpus. "
                     "Provide a metadata map with titles/arXiv IDs for the CorpusIDs."
        }, eval_results

    recall = hits_count / total_with_gold
    mrr = rr_sum / total_with_gold
    hit1 = hits1 / total_with_gold

    summary = {
        "queries_total": len(bench),
        "queries_with_mapped_golds": total_with_gold,
        "k": k,
        "recall@k": recall,
        "hits@1": hit1,
        "mrr": mrr,
    }
    return summary, eval_results


# --------------------------
# CLI
# --------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on AstaBench PaperFindingBench.")
    parser.add_argument(
        "--bench",
        type=Path,
        default=Path(".tmp_astabench/astabench/evals/paper_finder/data.json"),
        help="Path to AstaBench PaperFindingBench JSON (dev or test).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/database.parquet"),
        help="Local corpus parquet used to build the index.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/arxiv_nlp.csv"),
        help="Fallback CSV corpus.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        help="Optional metadata JSON/JSONL mapping CorpusID -> {title, externalIds{ArXiv, DOI}}.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Optional random sample size from the benchmark.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k cutoff for recall / MRR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the summary JSON.",
    )
    args = parser.parse_args()

    if not wait_for_index_ready(timeout_s=1.0):
        raise SystemExit(
            "Elasticsearch index is empty. Run `docker compose up -d elasticsearch` "
            "and `REBUILD=1 python scripts/process_db.py` first."
        )

    summary, results = run_eval(
        bench_path=args.bench,
        parquet_path=args.parquet,
        csv_path=args.csv,
        meta_path=args.meta,
        sample=args.sample,
        seed=args.seed,
        k=args.k,
    )

    print("\n--- AstaBench PaperFinder (local recall proxy) ---")
    if "error" in summary:
        print(summary["error"])
    else:
        print(f"Queries (with mapped golds): {summary['queries_with_mapped_golds']}/{summary['queries_total']}")
        print(f"Recall@{args.k}: {summary['recall@k']:.3f}")
        print(f"Hits@1: {summary['hits@1']:.3f}")
        print(f"MRR: {summary['mrr']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "summary": summary,
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "gold_local_ids": sorted(r.gold_local_ids),
                    "found_rank": r.found_rank,
                    "top_hits": [
                        {
                            "paper_id": h.get("paper_id"),
                            "title": h.get("title"),
                            "score": h.get("final_score") or h.get("score"),
                        }
                        for h in r.top_hits
                    ],
                }
                for r in results
            ],
        }, indent=2), encoding="utf-8")
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
