"""
Lightweight retrieval benchmark for the Asta paper finder.

It uses your existing Elasticsearch index and treats each paper's title as a
query; the target document is the same paper ID. Metrics: Recall@k, MRR, and
Hits@1. This avoids LLM calls and runs quickly on the current corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from tqdm import tqdm

from scripts.backend.services.retrieval.elastic_client import wait_for_index_ready
from scripts.backend.services.retrieval.retriever import run_vector_search


def _read_frame(parquet_path: Path, csv_path: Path) -> pd.DataFrame:
    """
    Load the smallest available frame; prefer parquet for speed.
    """
    if parquet_path.exists():
        try:
            df = pd.read_parquet(
                parquet_path,
                columns=["paper_id", "id", "title", "abstract"],
            )
        except Exception:
            df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Could not find a dataset at {parquet_path} or {csv_path}"
        )

    if "paper_id" not in df.columns:
        df["paper_id"] = df.get("id")

    df = df[["paper_id", "title", "abstract"]].dropna(subset=["title"])
    df["paper_id"] = df["paper_id"].astype(str)
    return df


def _build_queries(df: pd.DataFrame, sample: int, seed: int) -> List[Tuple[str, str]]:
    """
    Build (query, paper_id) pairs using the paper title as the query text.
    """
    rng = random.Random(seed)
    rows = df.sample(n=min(sample, len(df)), random_state=seed).itertuples()

    pairs: List[Tuple[str, str]] = []
    for r in rows:
        title = str(r.title).strip()
        if not title:
            continue
        pairs.append((title, str(r.paper_id)))

    rng.shuffle(pairs)
    return pairs


def _hit_rank(preds: Iterable[dict], target: str, k: int) -> int | None:
    """
    Return 1-based rank of the target paper_id in the top-k hits, or None.
    """
    for idx, hit in enumerate(list(preds)[:k]):
        if str(hit.get("paper_id")) == target:
            return idx + 1
    return None


def run_benchmark(pairs: List[Tuple[str, str]], k: int) -> dict:
    """
    Execute retrieval for each query and accumulate metrics.
    """
    total = len(pairs)
    if total == 0:
        return {"total": 0}

    hits = 0
    rr_sum = 0.0
    ranks_found: List[int] = []
    misses: List[dict] = []

    for query, target in tqdm(pairs, desc="Evaluating", unit="q"):
        results = run_vector_search(
            refined_query=query,
            k=k,
            analysis={},          # no filters/LLM; we only test retrieval
            show_scores=False,
        )
        rank = _hit_rank(results, target, k)
        if rank is None:
            misses.append({"query": query, "paper_id": target})
            continue
        hits += 1
        rr_sum += 1.0 / rank
        ranks_found.append(rank)

    recall = hits / total
    mrr = rr_sum / total
    hit_rate = sum(1 for r in ranks_found if r == 1) / total
    median_rank = float(pd.Series(ranks_found).median()) if ranks_found else None

    return {
        "total_queries": total,
        "k": k,
        "recall@k": recall,
        "mrr": mrr,
        "hits@1": hit_rate,
        "median_rank_when_found": median_rank,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick retrieval benchmark.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/database.parquet"),
        help="Path to the parquet dataset used to build the index.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/arxiv_nlp.csv"),
        help="Fallback CSV if parquet is missing.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Number of queries to evaluate (sampled).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Evaluate Recall/MRR at this cutoff.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for sampling/shuffling.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the raw metrics JSON.",
    )
    args = parser.parse_args()

    if not wait_for_index_ready(timeout_s=1.0):
        raise SystemExit(
            "Elasticsearch index is empty. Run `docker compose up -d elasticsearch` "
            "and `REBUILD=1 python scripts/process_db.py` first."
        )

    df = _read_frame(args.parquet, args.csv)
    pairs = _build_queries(df, sample=args.sample, seed=args.seed)
    metrics = run_benchmark(pairs, k=args.k)

    print("\n--- Retrieval Benchmark ---")
    print(f"Queries evaluated: {metrics.get('total_queries', 0)}")
    print(f"Recall@{args.k}: {metrics.get('recall@k', 0):.3f}")
    print(f"Hits@1: {metrics.get('hits@1', 0):.3f}")
    print(f"MRR: {metrics.get('mrr', 0):.3f}")
    if metrics.get("median_rank_when_found") is not None:
        print(f"Median rank when found: {metrics['median_rank_when_found']:.1f}")
    print(f"Misses (target not in top-{args.k}): {len(metrics.get('misses', []))}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics to {args.output}")


if __name__ == "__main__":
    main()
