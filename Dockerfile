# ---- Base image ----
FROM python:3.12-slim

# ---- Set working directory ----
WORKDIR /app

# ---- System deps ----
RUN apt-get update && apt-get install -y \
    build-essential curl git && rm -rf /var/lib/apt/lists/*

# ---- Copy project ----
COPY . .

# ---- Install dependencies ----
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# (Optional) If you don't have requirements.txt yet, you can inline packages:
# RUN pip install langchain langchain-community langchain-huggingface \
#     elasticsearch langchain-elasticsearch python-dotenv

# ---- Environment vars ----
ENV PYTHONUNBUFFERED=1
ENV CHROMA_PATH=/app/chroma
ENV PYTHONPATH=/app
# ---- Default command ----
CMD ["scripts/entrypoint.sh"]
