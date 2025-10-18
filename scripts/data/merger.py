import pandas as pd
from pathlib import Path

# ======================
# PATH CONFIGURATION
# ======================
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/Users/nihar/Documents/GitHub/AI-Research-Assistant-Agent/data")

OUT_DIR = DATA_DIR / "text_batches_db"

# ======================
# LOAD DATASETS
# ======================
meta = pd.read_parquet(DATA_DIR / "arxiv_cscl.parquet")
batches = pd.concat(
    [pd.read_parquet(f) for f in OUT_DIR.glob("*.parquet")],
    ignore_index=True
)

print(f"Loaded metadata: {meta.shape}, text batches: {batches.shape}")

# ======================
# EXTRACT ID (EXACT SAME AS DOWNLOADER)
# ======================
# arxiv_id = pdf_url.split("/")[-1].replace(".pdf", "")
meta["id"] = meta["pdf_url"].apply(lambda x: x.split("/")[-1].replace(".pdf", "") if isinstance(x, str) else None)

print("\nSample IDs from metadata:")
print(meta[["pdf_url", "id"]].head(10))

# ======================
# NORMALIZE BATCH COLUMN NAME
# ======================
if "arxiv_id" in batches.columns:
    batches = batches.rename(columns={"arxiv_id": "id"})
elif "paper_id" in batches.columns:
    batches = batches.rename(columns={"paper_id": "id"})

# ======================
# MERGE ON ID
# ======================
merged = meta.merge(batches, on="id", how="left")

# ======================
# SAVE MERGED DATASET
# ======================
merged.to_parquet(DATA_DIR / "arxiv_cscl_full.parquet", index=False)
merged.to_csv(DATA_DIR / "arxiv_cscl_full.csv", index=False)

print("\n✅ Merged successfully!")
print("Rows in merged file:", len(merged))
print("Missing text entries:", merged['text'].isna().sum())
