import argparse
import os
from pathlib import Path
from dotenv import load_dotenv
# from dataclasses import dataclass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

MODEL_ID = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

def generate_answer(context_text, question):
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.2,
        do_sample=False
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def build_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    es = Elasticsearch(ES_URL)
    store = ElasticsearchStore(
        index_name=ES_INDEX,
        embedding=embeddings,
        es_connection=es,
        es_url=ES_URL,
        vector_query_field="embedding",
    )
    return store

def format_sources(results_with_scores):
    out = []
    for doc, score in results_with_scores:
        meta = doc.metadata or {}
        out.append({
            "source": meta.get("source"),
            "row": meta.get("row"),
            "start_index": meta.get("start_index"),
            "score": float(score) if score is not None else None
        })
    return out

def main():
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query Elasticsearch index built with HuggingFace embeddings.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of results to retrieve (default: 3).")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score to *flag* low-quality (does not filter). Default 0.0.")
    parser.add_argument("--show-scores", action="store_true", help="Print similarity scores.")
    args = parser.parse_args()

    query_text = args.query_text
    store = build_store()

    # Prepare the DB.
    
    # Search the DB.
    results = store.similarity_search_with_relevance_scores(query_text, k=args.k)

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

    answer = generate_answer(context_text, query_text)
    print(answer)


if __name__ == "__main__":
    main()