# scripts/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from scripts.backend.query_reanalyzer_llm import reanalyze_query_llm
from scripts.backend.services.pipeline.rag_pipeline import query_rag
from scripts.backend.query_analyzer_llm import analyze_query_llm
from scripts.backend.llm_utils import MODEL_NAME
import inspect
import asyncio
import uuid

# mutex lock for searching (not yet used, but fine to keep)
_llm_lock = asyncio.Lock()
_active_searches: Dict[str, asyncio.Task] = {}
_active_searches_lock = asyncio.Lock()

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
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
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
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                set: list,
            }


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class Card(BaseModel):
    verdict: Optional[str] = None
    score: Optional[float] = None
    justification: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    facts: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    evidence_quotes: List[str] = Field(default_factory=list)
    score_note: Optional[str] = None

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
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
    card: Optional[Card] = None
    signals: PaperSignalsModel
    # FIX: typo in Field keyword
    evidence: List[Hit] = Field(default_factory=list)

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                set: list,
            }


class SearchResponse(BaseModel):
    query: str
    refined_query: Optional[str] = None
    top_score: Optional[float] = None
    hits: List[Hit] = Field(default_factory=list)
    papers: List[Paper] = Field(default_factory=list)
    context: str

    if _HAS_NP:
        class Config:
            json_encoders = {
                np.float32: float,
                np.float64: float,
                np.int32: int,
                np.int64: int,
                np.bool_: bool,
                np.ndarray: lambda a: a.tolist(),
                set: list,
            }


class SearchEnvelope(BaseModel):
    analysis: Any = None
    results: SearchResponse
    model: str
    reply_text: Optional[str] = None
# -----------------------------------------------------------------


class FeedbackRequest(BaseModel):
    query: str
    feedback: str
    analysis: Optional[dict] = None


class SearchRequest(BaseModel):
    # IMPORTANT: use 'query' if your frontend sends { "query": "..." }
    query: str
    k: int = Field(5, ge=1, le=100)
    search_id: Optional[str] = None
    show_scores: bool = True
    history: List[ChatMessage] = Field(default_factory=list)

    def ensure_search_id(self) -> str:
        if self.search_id:
            return self.search_id
        # auto-generate if frontend didn't send one
        self.search_id = uuid.uuid4().hex
        return self.search_id


async def _run_search_impl(req: "SearchRequest") -> SearchEnvelope:
    """
    Core search logic, extracted so it can be wrapped in a cancellable Task.
    This is basically your old /api/search body.
    """
    try:
        analysis = await call_maybe_async(analyze_query_llm, req.query)
    except Exception as e:
        print("[analyze_query_llm] failed:", e, flush=True)
        analysis = {"fallback": True, "error": str(e)}

    try:
        raw_results = await call_maybe_async(
            query_rag,
            req.query,
            k=req.k,
            show_scores=req.show_scores,
        )
    except Exception as e:
        print("[query_rag] failed:", e, flush=True)
        raw_results = {
            "query": req.query,
            "top_score": None,
            "hits": [],
            "papers": [],
            "context": "",
        }

    results = (
        raw_results
        if isinstance(raw_results, SearchResponse)
        else SearchResponse.parse_obj(raw_results)
    )

    reply_bits: List[str] = []
    reply_bits.append(f"Here are {len(results.papers)} papers for “{req.query}”.")
    if results.refined_query and results.refined_query != req.query:
        reply_bits.append(f"Refined query: “{results.refined_query}”.")
    if analysis.get("authors"):
        reply_bits.append(f"Author filter: {', '.join(analysis.get('authors', []))}.")
    if analysis.get("venues"):
        reply_bits.append(f"Venue filter: {', '.join(analysis.get('venues', []))}.")
    yr = analysis.get("time_range") or {}
    start, end = (yr.get("start"), yr.get("end")) if isinstance(yr, dict) else (None, None)
    if start or end:
        reply_bits.append(f"Year window: {start or 'any'}–{end or 'any'}.")
    reply_bits.append("Follow up with another question or refine your ask.")
    reply_text = " ".join(reply_bits)

    return SearchEnvelope(
        analysis=analysis,
        results=results,
        model=MODEL_NAME,
        reply_text=reply_text,
    )

# Optional: lightweight health checks
@app.get("/api/healthz")
def healthz():
    return {"ok": True}


# helper endpoint to prove which model/commit the backend is running
@app.get("/api/whoami")
def whoami():
    return {"model": MODEL_NAME}


@app.post("/api/search", response_model=SearchEnvelope)
async def search(req: SearchRequest):
    # ensure a search_id exists (for cancel + logging)
    search_id = req.ensure_search_id()

    async def _task_body():
        try:
            # run the actual search logic
            return await _run_search_impl(req)
        except asyncio.CancelledError:
            print(f"[search] cancelled: {search_id}", flush=True)
            # You can choose what to return if the client still waits:
            raise HTTPException(status_code=499, detail="Search cancelled")

    task = asyncio.create_task(_task_body(), name=f"search-{search_id}")

    # register the task
    async with _active_searches_lock:
        _active_searches[search_id] = task

    try:
        # wait for completion (or cancellation)
        return await task
    finally:
        # always clean up the registry entry
        async with _active_searches_lock:
            _active_searches.pop(search_id, None)

@app.post("/api/search/{search_id}/cancel")
async def cancel_search(search_id: str):
    async with _active_searches_lock:
        task = _active_searches.get(search_id)

    if task is None:
        # Either never existed, or already finished/cleaned up
        raise HTTPException(
            status_code=404,
            detail="Search not found or already completed",
        )

    # Ask asyncio to cancel the task
    task.cancel()
    return {"status": "cancelled", "search_id": search_id}

@app.post("/api/feedback")
async def handle_feedback(req: FeedbackRequest):
    # use attribute access for Pydantic models
    old_query = req.query
    feedback = req.feedback
    old_analysis = req.analysis

    # if these helpers might be async in future, use call_maybe_async
    updated = await call_maybe_async(
        reanalyze_query_llm,
        old_query,
        feedback,
        old_analysis,
    )
    result = await call_maybe_async(
        query_rag,
        updated["updated_query"],
        k=5,
        show_scores=True,
    )
    return {
        "new_query": updated["updated_query"],
        "changes": updated["changes"],
        "results": result,
    }
