# Asta Mock Up

# Running the Docker container
In order to run the Docker container, first start the Docker app. Then, if this is your first time running the project, run the following command in your terminal:
```
docker compose up -d elasticsearch
REBUILD=1 python scripts/process_db.py
```
Then regardless of whether or not it is your first time running the project, run the following in your terminal:
```
docker compose up --build
```

# Setting up the Database
First, run the database.py script to generate the .csv file containing papers from arXiv. Then, run process_db.py to process this data into the agent and train the embeddings. Note that these steps only need to be ran whenever you wish to fetch new papers from arXiv and update the training data of the RAG model.

# Starting up the Frontend
Start the frontend by first navigating to the frontend directory and then running
```
npm run dev
```