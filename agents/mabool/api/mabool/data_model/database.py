import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

import arxiv
import httpx
import fitz  # PyMuPDF
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
from dotenv import load_dotenv
import pyarrow

load_dotenv()

CATEGORIES = os.getenv("ARXIV_CATEGORIES", "cs.CL").split(",")
MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "5"))
DAYS_BACK = int(os.getenv("ARXIV_DAYS_BACK", "30"))
INCLUDE_PDF_TEXT = os.getenv("INCLUDE_PDF_TEXT", "true").lower() == "true"
OUT_PATH = os.getenv("OUT_PATH", "data/arxiv_nlp.parquet")

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

PDF_HEADERS = {"User-Agent": "arxiv-nlp-dataset/0.1 (+https://arxiv.org)"}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def download_pdf(url: str, timeout: int = 20) -> bytes:
    # return raw PDF bytes (not the Response obj)
    with httpx.Client(follow_redirects=True, timeout=timeout, headers=PDF_HEADERS) as client:
       try:
            r = client.get(url)
            r.raise_for_status()
       except httpx.HTTPStatusError: 
            return("Request Failed", httpx.HTTPStatusError) # type: ignore
       return r.content

def pdf_to_text(pdf_bytes: bytes) -> str:
    text_chunks = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            # be explicit about the extractor
            txt = page.get_text("text") or "" # type: ignore
            if not txt.strip():
                txt = "\n".join(str(b) for b in (page.get_text("blocks") or [])) # type: ignore
            text_chunks.append(txt)
    return "\n".join(text_chunks).strip()

def record_to_row(result: arxiv.Result, text: Optional[str]) -> Dict:
    authors = "; ".join(a.name for a in result.authors) if result.authors else ""
    primary = result.primary_category if hasattr(result, "primary_category") else ""
    cats = " ".join(result.categories) if result.categories else ""

    submitted = result.published if result.published else None
    updated = result.updated if result.updated else submitted

    return {
        "arxiv_id": result.entry_id.split("/")[-1],
        "title": (result.title or "").strip(),
        "abstract": (result.summary or "").strip(),
        "authors": authors,
        "primary_category": primary,
        "categories": cats,
        "submitted_at": submitted,
        "updated_at": updated,
        "pdf_url": result.pdf_url,
        "doi": getattr(result, "doi", None),
        "comment": getattr(result, "comment", None),
        "journal_ref": getattr(result, "journal_ref", None),
        "version": getattr(result, "versions", [{}])[-1].get("version", None) if getattr(result, "versions", None) else None,
        "text": text,
    }

def search_query_for_category(cat: str, start: datetime) -> str:
    return f"cat:{cat}"

def ensure_aware_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def is_within_window(updated_at: Optional[datetime], start: datetime) -> bool:
    if not updated_at:
        return False
    updated_at = ensure_aware_utc(updated_at)
    start = ensure_aware_utc(start) # type: ignore
    return updated_at >= start # type: ignore

def fetch_category(cat: str, start_dt: datetime, max_results: int) -> List[Dict]:
    q = search_query_for_category(cat, start_dt)
    search = arxiv.Search(
        query=q,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # or LastUpdatedDate
        sort_order=arxiv.SortOrder.Descending,
    )
    rows: List[Dict] = []
    try:
        for result in tqdm(search.results(), desc=f"Fetching {cat}", unit="paper"):
            updated = result.updated if result.updated else result.published
            if not is_within_window(updated, start_dt):
                continue

            text = None
            if INCLUDE_PDF_TEXT and result.pdf_url:
                try:
                    pdf_bytes = download_pdf(result.pdf_url)
                    text = pdf_to_text(pdf_bytes)
                    if not text:
                        # keep None if extraction produced empty string
                        text = None
                except Exception:
                    text = None

            rows.append(record_to_row(result, text))
    except arxiv.UnexpectedEmptyPageError:
        # treat as end-of-feed; return what we have
        pass
    return rows  # <-- always return rows

def main():
    start_dt = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    all_rows: List[Dict] = []
    seen_ids = set()

    for cat in CATEGORIES:
        rows = fetch_category(cat.strip(), start_dt, MAX_RESULTS)
        for r in rows:
            if r["arxiv_id"] not in seen_ids:
                seen_ids.add(r["arxiv_id"])
                all_rows.append(r)

    if not all_rows:
        print("No rows fetched. Try increasing DAYS_BACK or MAX_RESULTS.")
        return

    df = pd.DataFrame(all_rows).sort_values("updated_at", ascending=False).reset_index(drop=True)

    # Save only CSV (simplest + avoids parquet deps)
    csv_out = "data/arxiv_nlp.csv"
    parquet_out = "data/arxiv_nlp.parquet"
    saved = False
    try:
        df.to_parquet(parquet_out, index=False, engine="pyarrow", compression="snappy")
        print(f"Saved Parquet (pyarrow): {parquet_out}")
        saved = True
    except Exception as e1:
        print(f"[WARN] pyarrow failed: {e1}")
    
    if not saved:
        try:
            df.to_parquet(parquet_out, index=False, engine="fastparquet", compression="snappy")
            print(f"Saved Parquet (fastparquet): {parquet_out}")
            saved = True
        except Exception as e2:
            print(f"[WARN] fastparquet failed: {e2}")

    if not saved:
        df.to_csv(csv_out, index=False)
        print(f"Saved CSV (fallback): {csv_out}")


    df.to_csv(csv_out, index=False)
    print(f"Saved {len(df)} rows to:\n  {csv_out}")

if __name__ == "__main__":
    main()
