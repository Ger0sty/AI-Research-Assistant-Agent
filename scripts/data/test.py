import pandas as pd 
database = pd.read_parquet("/Users/nihar/Documents/GitHub/AI-Research-Assistant-Agent/data/arxiv_cscl_full.parquet")

print(database[database["status"] == "success"])