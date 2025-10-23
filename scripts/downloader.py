import asyncio
import aiohttp
import aiofiles
import pandas as pd
import fitz  # PyMuPDF
import io, random, os, json
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio

# ======================
# CONFIGURATION
# ======================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PARQUET_PATH = DATA_DIR / "arxiv_cscl.parquet"
BATCH_SIZE = 500
MAX_CONCURRENCY = 5                # safe for arXiv (slower, but steady)
TIMEOUT = aiohttp.ClientTimeout(total=120)
RETRY_LIMIT = 4
USER_AGENT = "Mozilla/5.0 (compatible; arxiv-batch-stream/2.0; email=YOUR_EMAIL@illinois.edu)"
PDF_TMP_DIR = DATA_DIR / "temp_pdfs"
OUT_DIR = DATA_DIR / "text_batches_db"
CHECKPOINT_FILE = DATA_DIR / "completed_batches.json"

PDF_TMP_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
headers = {"User-Agent": USER_AGENT}

# ======================
# LOAD / RESUME STATE
# ======================
completed_batches = set()
if CHECKPOINT_FILE.exists():
    with open(CHECKPOINT_FILE, "r") as f:
        completed_batches = set(json.load(f))
    print(f"✅ Resuming. {len(completed_batches)} batches already completed.")


# ======================
# HELPERS
# ======================
async def download_pdf(session, pdf_url):
    """Download one PDF with retries and polite backoff."""
    pdf_url = pdf_url.strip()
    if not pdf_url.endswith(".pdf"):
        pdf_url += ".pdf"
    arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
    pdf_path = PDF_TMP_DIR / f"{arxiv_id}.pdf"

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            async with session.get(pdf_url, headers=headers) as resp:
                # success
                if resp.status == 200 and "pdf" in resp.headers.get("Content-Type", ""):
                    async with aiofiles.open(pdf_path, "wb") as f:
                        await f.write(await resp.read())
                    return arxiv_id, pdf_path, "downloaded"

                # rate limited
                elif resp.status in (403, 429, 503):
                    wait = 60 * attempt
                    print(f"⏳ {arxiv_id}: rate-limited ({resp.status}), waiting {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                # other non-PDF response
                else:
                    return arxiv_id, pdf_path, f"bad_status_{resp.status}"

        except Exception as e:
            wait = 15 * attempt
            print(f"⚠️ {arxiv_id}: {type(e).__name__}, retrying in {wait}s...")
            await asyncio.sleep(wait)
    return arxiv_id, pdf_path, "failed"


def extract_text_from_pdf(pdf_path):
    """Extract text and clean up the temporary PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc) # type: ignore
        doc.close()
        os.remove(pdf_path)
        return text
    except Exception:
        if pdf_path.exists():
            os.remove(pdf_path)
        return ""


async def process_batch(session, batch_df, batch_index):
    """Handles one batch: download → extract → save → delete."""
    print(f"\n🚀 Processing batch {batch_index+1} ({len(batch_df)} papers)...")

    tasks = [download_pdf(session, url) for url in batch_df["pdf_url"]]
    download_results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        download_results.append(await coro)

    # Extract text + save
    texts = []
    for arxiv_id, pdf_path, status in download_results:
        if status == "downloaded":
            text = extract_text_from_pdf(pdf_path)
            texts.append({"id": arxiv_id, "text": text, "status": "success"})
        else:
            texts.append({"id": arxiv_id, "text": "", "status": status})

    out_path = OUT_DIR / f"text_batch_{(batch_index+1)*BATCH_SIZE}.parquet"
    pd.DataFrame(texts).to_parquet(out_path)
    print(f"💾 Saved batch {batch_index+1} → {out_path}")

    # mark this batch as done
    completed_batches.add(batch_index)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(list(completed_batches)), f)


# ======================
# MAIN PIPELINE
# ======================
async def main():
    df = pd.read_parquet(PARQUET_PATH)
    total = len(df)
    batches = [df[i:i+BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector, timeout=TIMEOUT) as session:
        for batch_index, batch_df in enumerate(batches):
            if batch_index in completed_batches:
                print(f"⏭️ Skipping batch {batch_index+1} (already done).")
                continue

            await process_batch(session, batch_df, batch_index)

            # polite cooldown after each batch (adaptive sleep)
            cool_down = random.uniform(90, 150)
            print(f"🛌 Cooling down for {cool_down:.1f}s before next batch...")
            await asyncio.sleep(cool_down)

    print("✅ All batches processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())