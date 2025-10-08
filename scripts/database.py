from pathlib import Path
from sickle import Sickle
import pandas as pd

# Declare Path variables
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

sickle = Sickle("https://oaipmh.arxiv.org/oai")
records = sickle.ListRecords(metadataPrefix="oai_dc", set="cs:cs:CL")


# This block of code below is essentially used to see what was being returned by OAI
"""for i, record in enumerate(records):
    print(record.metadata.keys())
    print("Identifier:", record.header.identifier)
    print(record.metadata.get("identifier"))
    print("Titles:", " | ".join(record.metadata.get("title", [])))
    print("Creators:", record.metadata.get("creator"))
    print("Subjects:", record.metadata.get("subject"))  # <- categories / keywords
    print("Abstract: ", record.metadata.get("description"))
    print("Date:", record.metadata.get("date"))
    print("Type:", record.metadata.get("type"))
    #print("=" * 80)

    if i > 0:  # stop after a few
        break"""

# This block of code below helped us identify the current notation of the category
"""sets = sickle.ListSets()
for i, s in enumerate(sets):
    print(s.setSpec, "-", s.setName)
    if i > 400:  # just print the first few
        break"""

data = []
for i, record in enumerate(records): 
    title = " | ".join(record.metadata.get("title", []))
    authors = " ; ".join(record.metadata.get("creator", []))
    category = " | ".join(record.metadata.get("subject", []))
    abstract = " | ".join(record.metadata.get("description", []))
    date = " | ".join(record.metadata.get("date", []))
    type = " | ".join(record.metadata.get("type", []))
    OAI_identifier = record.header.identifier
    abs_url = record.metadata.get("identifier", [""])[0]
    pdf_url = abs_url.replace("/abs/", "/pdf/")

    data.append([title, authors, category, abstract, date, type, OAI_identifier, abs_url, pdf_url ])

columns_initial = ['title', 'authors', 'category', 'abstract', 'date', 'type', 'OAI_identifier', 'abs_url', 'pdf_url']
database = pd.DataFrame(data, columns=columns_initial)
print(database)

database.to_csv(DATA_DIR / "arxiv_cscl.csv", index=False)
database.to_parquet(DATA_DIR / "arxiv_cscl.parquet", index=False)