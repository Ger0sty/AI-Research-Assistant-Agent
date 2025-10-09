# Asta Mock Up

# Running the Docker container
In order to run the Docker container, first start the Docker app. Then, run the following command in your terminal:
```
docker compose up --build
```

# Setting up the Database
First, run the database.py script to generate the .csv file containing papers from arXiv. Then, run process_db.py to process this data into the agent and train the embeddings. Note that these steps only need to be ran whenever you wish to fetch new papers from arXiv and update the training data of the RAG model.

# Querying the Model
Each time you wish to query the RAG model, you run the query_data.py script. There are two different ways to do this. The first is to make sure that you have built your Docker image, and then run outside Docker in your Mac terminal
```
docker compose run app python3 scripts/query_data.py {query_text} [args]
```
where query_text is the user's query inside of quotation marks, and args are the optional arguments that can be provided to the function. Note that some users may need to replace "python3" with "python". 

The second option for running the script is to run it within the container shell as follows:
```
docker exec -it asta-paper-finder-app-1 bash
python3 scripts/query_data.py {query_text} [args]
```
The different optional arguments are as follows:
* ``` --k ``` is the number of results returned, with a default value of 3
* ``` --min-score ``` is the minimum score (0-1) the model requires to consider the paper in the output, with a default value of 0.0
* ``` --show-scores ``` is an option that includes the similarity scores of in the printed output

Ex:
```
docker compose run app python3 scripts/query_data.py "Explain transformer models" --k 3 --show-scores
```