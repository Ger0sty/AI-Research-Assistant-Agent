# scripts/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from scripts.backend.query_reanalyzer_llm import reanalyze_query_llm
from scripts.backend.services.pipeline.rag_pipeline import query_rag
from scripts.backend.query_analyzer_llm import analyze_query_llm
from scripts.backend.llm_utils import MODEL_NAME
import inspect, asyncio
# mutex lock for searching
_llm_lock = asyncio.Lock()

async def call_maybe_async(fn, *args, **kwargs):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    return await asyncio.to_thread(fn, *args, **kwargs)

try:
    import numpy as np
    _HAS_NP = True
except Exception:
    _HAS_NP = False

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

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float, np.float64: float,
                np.int32: int, np.int64: int,
                np.bool_: bool, np.ndarray: lambda a: a.tolist(),
                set: list,
            }

class PaperSignalsModel(BaseModel):
    max_score: float
    mean_score: float
    coverage: float
    over_threshold: float
    query_overlap_terms: List[str]
    author_matched: bool
    venue_boost: float
    recency_boost: float

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float, np.float64: float,
                np.int32: int, np.int64: int,
                np.bool_: bool, np.ndarray: lambda a: a.tolist(),
                set: list,
            }

class Paper(BaseModel):
    paper_id: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None
    explanation: Optional[str] = None
    signals: PaperSignalsModel
    evidence: List[Hit] = Field(dfault_factory=list)

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float, np.float64: float,
                np.int32: int, np.int64: int,
                np.bool_: bool, np.ndarray: lambda a: a.tolist(),
                set: list,
            }

class SearchResponse(BaseModel):
    query: str
    top_score: Optional[float] = None
    hits: List[Hit] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)
    context: str

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float, np.float64: float,
                np.int32: int, np.int64: int,
                np.bool_: bool, np.ndarray: lambda a: a.tolist(),
                set: list,
            }

class SearchEnvelope(BaseModel):
    analysis: Any = None
    results: SearchResponse
    model: str
# -----------------------------------------------------------------

class FeedbackRequest(BaseModel):
    query: str
    feedback: str
    analysis: Optional[dict] = None

class SearchRequest(BaseModel):
    q: str
    k: int = Field(5, ge=1, le=100)
    show_scores: bool = True

# Optional: lightweight health checks
@app.get("/api/healthz")
def healthz():
    return {"ok": True}

# NEW: helper endpoint to prove which model/commit the backend is running
@app.get("/api/whoami")  # NEW
def whoami():
    from scripts.backend.llm_utils import MODEL_NAME
    return {"model": MODEL_NAME}                # NEW

@app.post("/api/search", response_model=SearchEnvelope)
async def search(req: SearchRequest):
    try: 
        analysis = await call_maybe_async(analyze_query_llm, req.q)
    except Exception as e:
        print("[analyze_query_llm] failed:", e, flush=True)
        analysis = {"fallback": True, "error": str(e)}
    
    try: 
        raw_results = await call_maybe_async(query_rag, req.q, k=req.k, show_scores=req.show_scores)
    except Exception as e:
        print("[query_rag] failed:", e, flush=True)
        raw_results = {"query": req.q, "top_score": None, "hits": [], "papers": [], "context": ""}
    
    results = (
    raw_results
    if isinstance(raw_results, SearchResponse)
    else SearchResponse.parse_obj(raw_results)
    )

    return SearchEnvelope(
        analysis=analysis,
        results=results,
        model=MODEL_NAME,
    )

@app.post("/api/feedback")
def handle_feedback(req: FeedbackRequest):
    old_query = req["query"]
    feedback = req["feedback"]
    old_analysis = req["analysis"]
    updated = reanalyze_query_llm(old_query, feedback, old_analysis)
    result = query_rag(updated["updated_query"], k=5)
    return {"new_query": updated["updated_query"], "changes": updated["changes"], "results": result}
