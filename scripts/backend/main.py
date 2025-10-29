# scripts/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

from scripts.backend.rag_service import query_rag, wait_for_index_ready

app = FastAPI()

# Dev CORS (tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

# --------- Pydantic models (match rag_service.py output) ---------
class Hit(BaseModel):
    content: str
    score: Optional[float] = None
    display_score: Optional[float] = None
    source: Optional[str] = None
    row: Optional[int] = None
    start_index: Optional[int] = None
    paper_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None

class PaperSignalsModel(BaseModel):
    max_score: float
    mean_score: float
    coverage: int
    over_threshold: int
    query_overlap_terms: List[str]
    author_matched: bool
    venue_boost: float
    recency_boost: float

class Paper(BaseModel):
    paper_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    explanation: Optional[str] = None
    signals: PaperSignalsModel
    evidence: List[Hit]

class SearchResponse(BaseModel):
    query: str
    top_score: Optional[float] = None
    hits: List[Hit]
    papers: List[Paper]
    context: str
# -----------------------------------------------------------------

class SearchRequest(BaseModel):
    q: str
    k: int = Field(5, ge=1, le=100)
    show_scores: bool = True

@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not wait_for_index_ready(timeout_s=0):
        raise HTTPException(status_code=503, detail="Index is building. Try again in a few seconds.")
    try:
        print(f"[search] → query='{req.q}' k={req.k} show_scores={req.show_scores}")
        result = query_rag(req.q, k=req.k, show_scores=req.show_scores)
        print(f"[search] ✓ returned {len(result.get('papers', []))} papers")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()  # <--- prints the real error to logs
        raise HTTPException(status_code=500, detail=str(e))
