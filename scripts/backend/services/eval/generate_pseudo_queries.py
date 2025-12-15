"""
Generate in-domain pseudo-benchmark queries from your existing corpus
without using an LLM. Outputs JSONL with fields:
  {"query": "...", "paper_id": "...", "source": "..."}

Strategies:
  - Title-based: use the paper title directly.
  - Keyword-based: pick top keywords from the abstract and build a short query.
You can append your own curated queries to the output file to improve quality.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "to",
    "in",
    "for",
    "on",
    "with",
    "by",
    "from",
    "using",
    "based",
    "approach",
    "method",
    "models",
    "model",
    "paper",
    "results",
    "study",
}


def _read_corpus(parquet_path: Path, csv_path: Path, sample: int, seed: int) -> pd.DataFrame:
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No corpus found at {parquet_path} or {csv_path}")

    if "paper_id" not in df.columns:
        df["paper_id"] = df.get("id")

    keep_cols = [c for c in ["paper_id", "title", "abstract"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["title"])
    df = df.sample(n=min(sample, len(df)), random_state=seed)
    df["paper_id"] = df["paper_id"].astype(str)
    return df


def _top_keywords(text: str, max_k: int = 5) -> List[str]:
    toks = re.findall(r"[a-zA-Z0-9]{3,}", text.lower())
    toks = [t for t in toks if t not in STOPWORDS]
    counts = Counter(toks)
    return [w for w, _ in counts.most_common(max_k)]


def _title_query(title: str) -> str:
    return title.strip()


def _keyword_query(title: str, abstract: str) -> str | None:
    kws = _top_keywords(title + " " + abstract, max_k=4)
    if not kws:
        return None
    if len(kws) == 1:
        return f"papers about {kws[0]}"
    return "papers about " + ", ".join(kws[:2])


def generate_queries(df: pd.DataFrame) -> List[dict]:
    out: List[dict] = []
    for row in df.itertuples():
        pid = row.paper_id
        title = getattr(row, "title", "") or ""
        abstract = getattr(row, "abstract", "") or ""

        # title-based
        tq = _title_query(title)
        if tq:
            out.append({"query": tq, "paper_id": pid, "source": "title"})

        # keyword-based
        kq = _keyword_query(title, abstract)
        if kq:
            out.append({"query": kq, "paper_id": pid, "source": "keywords"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pseudo-queries from the local corpus.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/database.parquet"),
        help="Parquet corpus path.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/arxiv_nlp.csv"),
        help="CSV corpus path (fallback).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=300,
        help="Number of documents to sample for query generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmarks/pseudo_queries.jsonl"),
        help="Where to write the JSONL queries.",
    )
    args = parser.parse_args()

    df = _read_corpus(args.parquet, args.csv, sample=args.sample, seed=args.seed)
    queries = generate_queries(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for obj in queries:
            f.write(json.dumps(obj) + "\n")

    print(f"Wrote {len(queries)} queries to {args.output}")
    print("Edit or append your own curated queries to this file as needed.")


if __name__ == "__main__":
    main()
