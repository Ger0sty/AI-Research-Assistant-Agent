# backend/main.py
from fastapi import FastAPI, HTTPException
from scripts.backend.rag_service import query_rag, wait_for_index_ready
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .rag_service import query_rag

app = FastAPI()

# Dev CORS (fine because Vite will proxy; keep or restrict domains in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}


class SearchRequest(BaseModel):
    q: str
    k: int = Field(5, ge=1, le=100) 
    show_scores: bool = True

@app.post("/api/search")
def search(req: SearchRequest):
    # quick check: is the index ready?
    if not wait_for_index_ready(timeout_s=0):
        # optional: include Retry-After so your frontend can auto-retry
        raise HTTPException(status_code=503, detail="Index is building. Try again in a few seconds.")
    try:
        return query_rag(req.q, k=req.k, show_scores=req.show_scores)
    except Exception as e:
        # never let the worker crash; return 500 with message
        raise HTTPException(status_code=500, detail=str(e))
