#!/bin/sh
set -eu

ES_URL="${ES_URL:-http://elasticsearch:9200}"
ES_INDEX="${ES_INDEX:-rag_docs}"

echo "⏳ Waiting for Elasticsearch at $ES_URL ..."
until curl -sf "$ES_URL" >/dev/null 2>&1; do
  sleep 2
done
echo "✅ Elasticsearch is up."

# Indexing policy
if [ "${REBUILD:-0}" = "1" ]; then
  echo "🔄 REBUILD=1 → indexing in background..."
  python scripts/process_db.py &
else
  if curl -sfI "$ES_URL/$ES_INDEX" >/dev/null 2>&1; then
    echo "✅ Index '$ES_INDEX' exists → skipping indexing."
  else
    echo "📚 Index '$ES_INDEX' missing → indexing in background..."
    python scripts/process_db.py &
  fi
fi

echo "🚀 Starting FastAPI backend..."
exec uvicorn scripts.backend.main:app --host 0.0.0.0 --port 8000
