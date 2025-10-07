import pymupdf
import pandas as pd
import requests
import fitz
import numpy as np

database = pd.read_parquet("arxiv_cscl.parquet")
database["pdf_text"] = pd.Series(dtype="object")

count = 0
for i, url in enumerate(database["pdf_url"]):
    response = requests.get(url)
    doc = fitz.open(stream=response.content, filetype="pdf")
    text = ""
    count += 1
    for page in doc:
        text += page.get_text() # type: ignore
    database.at[i, "pdf_text"] = text
    if count == 100: 
        break



print(database["pdf_text"])
database.to_parquet("arxiv_cscl_with_text.parquet")
database.to_csv("arxiv_cscl_with_text.csv")
