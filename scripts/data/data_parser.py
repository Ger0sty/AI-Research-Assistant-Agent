from pathlib import Path
import pandas as pd
import requests
import fitz
import numpy as np

# Declare paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IN_PATH = DATA_DIR / "arxiv_cscl.parquet"
OUT_PARQUET = DATA_DIR / "arxiv_cscl_with_text.parquet"
OUT_CSV = DATA_DIR / "arxiv_cscl_with_text.csv"

database = pd.read_parquet(IN_PATH)
if "pdf_text" not in database.columns:
    database["pdf_text"] = pd.Series(index=database.index, dtype="object")

count = 0
for i, url in enumerate(database["pdf_url"]):
    response = requests.get(url)
    doc = fitz.open(stream=response.content, filetype="pdf")
    text = ""
    count += 1
    for page in doc:
        text += page.get_text() # type: ignore
    database.at[i, "pdf_text"] = text
    if count >= 100: 
        break



print(database["pdf_text"])
database.to_parquet(OUT_PARQUET, index=False)
database.to_csv(OUT_CSV, index=False)
