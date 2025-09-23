import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Protocol

import arxiv
import httpx
import pymupdf as fitz  # use pymupdf (module name) to avoid name clashes
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ------- Config -------
CATEGORIES = os.getenv("ARXIV_CATEGORIES", "cs.CL").split(",")
DAYS_BACK = int(os.getenv("ARXIV_DAYS_BACK", "1"))
INCLUDE_PDF_TEXT = os.getenv("INCLUDE_PDF_TEXT", "true").lower() == "true"
OUT_PATH = os.getenv("OUT_PATH", "data/arxiv_nlp.parquet")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

PDF_HEADERS = {"User-Agent": "arxiv-nlp-dataset/0.1 (+https://arxiv.org)"}


# ------- Networking -------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def download_pdf(url: str, timeout: int = 20) -> bytes:
    """Return raw PDF bytes; raise on failure or non-PDF responses."""
    with httpx.Client(
        follow_redirects=True, timeout=timeout, headers=PDF_HEADERS
    ) as client:
        r = client.get(url)
        r.raise_for_status()
        # Ensure we didn't fetch HTML or something else
        ct = (r.headers.get("content-type") or "").lower()
        if not r.content.startswith(b"%PDF-") and "pdf" not in ct:
            raise ValueError(f"Not a PDF: {url} (content-type={ct})")
        return r.content


# ------- PDF Text Extraction (Pylance-friendly) -------
class _Page(Protocol):
    # Minimal protocol so Pylance knows the object has get_text(...)
    def get_text(self, option: str = "text"):
        ...


def pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes; fallback to 'blocks' when plain text is empty."""
    chunks: List[str] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        # if encrypted and empty password doesn't work, bail
        if doc.is_encrypted and not doc.authenticate(""):
            return ""

        for i in range(doc.page_count):
            page: _Page = doc.load_page(i)  # type: ignore # tell the type-checker this has get_text()
            txt = page.get_text("text") or ""
            if not txt.strip():
                blk = page.get_text("blocks") or []
                if blk:
                    txt = "\n".join(str(b) for b in blk)
            chunks.append(txt.strip())

        return "\n\n".join(s for s in chunks if s)
    finally:
        doc.close()


# ------- Row construction -------
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
        "version": getattr(result, "versions", [{}])[-1].get("version", None)
        if getattr(result, "versions", None)
        else None,
        "text": text,
    }


def search_query_for_category(cat: str, start: datetime) -> str:
    # Keep simple; date filtering handled by is_within_window
    return f"cat:{cat}"


# ------- Time helpers (typed for Optional) -------
def ensure_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_within_window(updated_at: Optional[datetime], start: datetime) -> bool:
    if updated_at is None:
        return False
    ua = ensure_aware_utc(updated_at)
    sa = ensure_aware_utc(start)
    assert ua is not None and sa is not None
    return ua >= sa


# ------- Fetch loop -------
def fetch_category(cat: str, start_dt: datetime) -> List[Dict]:
    q = search_query_for_category(cat, start_dt)
    search = arxiv.Search(
        query=q,
        sort_by=arxiv.SortCriterion.SubmittedDate,  # or LastUpdatedDate
        sort_order=arxiv.SortOrder.Descending,
    )
    rows: List[Dict] = []
    try:
        for result in tqdm(search.results(), desc=f"Fetching {cat}", unit="paper"):
            updated = result.updated if result.updated else result.published
            if not is_within_window(updated, start_dt):
                continue

            text: Optional[str] = None
            if INCLUDE_PDF_TEXT and result.pdf_url:
                try:
                    pdf_bytes = download_pdf(result.pdf_url)
                    text = pdf_to_text(pdf_bytes) or None
                except (httpx.HTTPError, ValueError, Exception):
                    text = None

            rows.append(record_to_row(result, text))
    except arxiv.UnexpectedEmptyPageError:
        pass  # end-of-feed
    return rows


# ------- Main -------
def main():
    start_dt = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)
    all_rows: List[Dict] = []
    seen_ids = set()

    for cat in CATEGORIES:
        rows = fetch_category(cat.strip(), start_dt)
        for r in rows:
            if r["arxiv_id"] not in seen_ids:
                seen_ids.add(r["arxiv_id"])
                all_rows.append(r)

    if not all_rows:
        print("No rows fetched. Try increasing DAYS_BACK or adjusting CATEGORIES.")
        return

    df = (
        pd.DataFrame(all_rows)
        .sort_values("updated_at", ascending=False)
        .reset_index(drop=True)
    )

    csv_out = "data/arxiv_nlp.csv"
    parquet_out = "data/arxiv_nlp.parquet"
    saved_parquet = False

    # Try pyarrow first
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(parquet_out, index=False, engine="pyarrow", compression="snappy")
        print(f"Saved Parquet (pyarrow): {parquet_out}")
        saved_parquet = True
    except Exception as e1:
        print(f"[WARN] pyarrow failed: {e1}")

    # Fallback to fastparquet
    if not saved_parquet:
        try:
            import fastparquet  # noqa: F401
            df.to_parquet(
                parquet_out, index=False, engine="fastparquet", compression="snappy"
            )
            print(f"Saved Parquet (fastparquet): {parquet_out}")
            saved_parquet = True
        except Exception as e2:
            print(f"[WARN] fastparquet failed: {e2}")

    # Final fallback to CSV
    if not saved_parquet:
        df.to_csv(csv_out, index=False)
        print(f"Saved CSV (fallback): {csv_out}")

    print(f"Saved {len(df)} rows to: {parquet_out if saved_parquet else csv_out}")


if __name__ == "__main__":
    main()
