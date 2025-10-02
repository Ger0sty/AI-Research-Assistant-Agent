import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

def build_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory = str(ROOT_DIR / CHROMA_PATH),
        embedding_function = embeddings,
    )
    return db

def format_sources(results):
    sources = []
    for doc, score in results:
        meta = doc.metadata or {}
        src = meta.get("source")
        row = meta.get("row")
        start = meta.get("start_index")
        sources.append({"source":src, "row":row, "start_index":start, "score":float(score) if score is not None else None,})
    return sources

def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query local Chroma index built with HuggingFace embeddings.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve (default: 3).")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score to *flag* low-quality (does not filter). Default 0.0.")
    parser.add_argument("--show-scores", action="store_true", help="Print similarity scores.")
    args = parser.parse_args()

    query_text = args.query_text
    db = build_db()

    # Prepare the DB.
    
    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=args.k)

    if not results:
        print("No results returned. (Empty index or path/embedding mismatch.)")
        return

    # Always show the top-k results; warn if low scores
    if args.show_scores:
        print("=== TOP RESULTS ===")
        for i, (doc, score) in enumerate(results, 1):
            print(f"{i}. score={score:.3f}  |  {doc.metadata}")

    # Flag (but don't suppress) low scoring hits
    top_score = results[0][1]
    if top_score < args.min_score:
        print(f"\n⚠️  Top score {top_score:.3f} is below --min-score {args.min_score:.3f} (treat as low-confidence).")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
   
    print(context_text)

    print("\n=== SOURCES ===")
    for i, entry in enumerate(format_sources(results), start=1):
        line = f"{i}. {entry['source']} (row={entry['row']}, start_index={entry['start_index']})"
        if args.show_scores:
            line += f" | score={entry['score']:.3f}"
        print(line)


if __name__ == "__main__":
    main()