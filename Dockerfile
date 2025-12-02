# ---- Base image ----
FROM python:3.12-slim

# ---- Set working directory ----
WORKDIR /app

# ---- System deps needed for HF, ES client, vector stores ----
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    libffi-dev \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps first (caches better) ----
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Copy source code ----
COPY . .

# ---- Ensure entrypoint is executable ----
RUN chmod +x scripts/entrypoint.sh

# ---- Python environment ----
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CHROMA_PATH=/app/chroma

# (Optional defaults, docker-compose can override)
ENV ES_URL=http://elasticsearch:9200
ENV ES_INDEX=rag_docs

# ---- Default command ----
CMD ["bash", "scripts/entrypoint.sh"]
