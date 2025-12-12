# scripts/backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from scripts.backend.query_reanalyzer_llm import reanalyze_query_llm
from scripts.backend.services.pipeline.rag_pipeline import query_rag
from scripts.backend.query_analyzer_llm import analyze_query_llm
from scripts.backend.llm_utils import MODEL_NAME, call_llm_json, call_llm_json_last
import re
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


class PriorResults(BaseModel):
    last_query: Optional[str] = None
    last_refined_query: Optional[str] = None
    last_analysis: Optional[dict] = None
    last_papers: List[Paper] = Field(default_factory=list)


class SearchRequest(BaseModel):
    # IMPORTANT: use 'query' if your frontend sends { "query": "..." }
    query: str
    k: int = Field(5, ge=1, le=100)
    search_id: Optional[str] = None
    show_scores: bool = True
    history: List[ChatMessage] = Field(default_factory=list)
    prior_results: Optional[PriorResults] = None

    def ensure_search_id(self) -> str:
        if self.search_id:
            return self.search_id
        # auto-generate if frontend didn't send one
        self.search_id = uuid.uuid4().hex
        return self.search_id


_FOLLOWUP_HINTS = [
    "previous papers",
    "those papers",
    "these papers",
    "that paper",
    "second paper",
    "third paper",
    "first paper",
    "paper 2",
    "paper #2",
    "tell me more",
    "more info",
    "more information",
    "details on",
    "the above",
]

_ORDINAL_MAP = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
}

_AUTHOR_PHRASES = re.compile(r"\b(?:papers?|publications?|work)\s+by\s+", re.IGNORECASE)
_NAME_TOKEN = re.compile(r"[A-Z][A-Za-z\-\.'`]+(?:\s+[A-Z][A-Za-z\-\.'`]+)+", re.IGNORECASE)


def _shorten(txt: Optional[str], limit: int = 240) -> str:
    if not txt:
        return ""
    clean = re.sub(r"\s+", " ", str(txt)).strip()
    return clean[:limit] + ("…" if len(clean) > limit else "")


def _looks_like_author_query(q: str) -> bool:
    """
    Heuristic to preserve author searches: detect 'papers by <Name>'.
    """
    if _AUTHOR_PHRASES.search(q):
        return True
    lower_q = q.lower()
    if " by " in lower_q:
        # detect "by Andrew Ng" even if "papers" is omitted
        pos = lower_q.find(" by ")
        tail = q[pos + 4 :]  # skip the " by " token
        if _NAME_TOKEN.search(tail):
            return True
    return False


def _looks_like_followup(q: str) -> bool:
    s = q.lower()
    if any(h in s for h in _FOLLOWUP_HINTS):
        return True
    if re.search(r"\b(?:first|second|third|fourth|fifth)\s+paper\b", s):
        return True
    num_mentions = re.findall(r"\bpaper\s*(?:#|number\s*)?(\d+)\b", s)
    if any(m.isdigit() and int(m) <= 10 for m in num_mentions):
        return True
    return False


def _indices_from_text(q: str, total: int) -> List[int]:
    s = q.lower()
    out: List[int] = []
    for word, idx in _ORDINAL_MAP.items():
        if word in s:
            out.append(idx)
    for m in re.findall(r"\bpaper\s*(?:#|number\s*)?(\d+)\b", s):
        try:
            val = int(m) - 1
            if 0 <= val < total:
                out.append(val)
        except Exception:
            continue
    for m in re.findall(r"\b(\d+)(?:st|nd|rd|th)\b", s):
        try:
            val = int(m) - 1
            if 0 <= val < total:
                out.append(val)
        except Exception:
            continue
    deduped = []
    seen = set()
    for i in out:
        if 0 <= i < total and i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


def _title_hits(q: str, papers: List[Paper]) -> List[int]:
    s = q.lower()
    hits: List[int] = []
    for idx, p in enumerate(papers):
        t = (p.title or "").lower()
        if not t:
            continue
        tokens = [tok for tok in re.findall(r"[a-z0-9]{4,}", t)]
        if tokens and any(tok in s for tok in tokens[:3]):
            hits.append(idx)
    return hits


def _select_paper_indices(q: str, papers: List[Paper]) -> List[int]:
    total = len(papers)
    if total == 0:
        return []
    idxs = _indices_from_text(q, total)
    if not idxs:
        idxs = _title_hits(q, papers)
    if not idxs:
        idxs = list(range(min(3, total)))
    deduped = []
    seen = set()
    for i in idxs:
        if 0 <= i < total and i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


def _paper_context_for_prompt(
    papers: List[Paper],
    idxs: List[int],
    followup_q: str = "",
    original_q: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ctx: List[Dict[str, Any]] = []
    for list_pos, paper_idx in enumerate(idxs, start=1):
        if paper_idx >= len(papers):
            continue
        p = papers[paper_idx]
        ev_snips: List[str] = []
        evidence_full: List[str] = []
        for ch in (p.evidence or [])[:3]:
            try:
                txt = ch.content
            except Exception:
                txt = ch.get("content") if isinstance(ch, dict) else None
            ev_snips.append(_shorten(txt))
            if txt:
                evidence_full.append(_shorten(txt, limit=360))
        ctx.append(
            {
                "rank": paper_idx + 1,
                "paper_id": p.paper_id,
                "title": p.title,
                "authors": p.authors,
                "venue": p.venue,
                "year": p.year,
                "url": p.url,
                "summary": (p.card.justification if p.card else None) or p.explanation,
                "evidence": ev_snips,
                "evidence_long": evidence_full,
                "original_query": original_q,
                "followup_query": followup_q,
            }
        )
    return ctx


def _llm_route_followup(user_q: str, prior: PriorResults, papers: List[Paper]) -> Dict[str, Any]:
    titles = [p.title for p in papers if p.title][:5]
    prompt = f"""
You route chat turns for a paper search assistant.
Decide if the user wants a NEW_SEARCH (retrieve new papers) or FOLLOWUP (answer using the previously returned papers).
- Pick FOLLOWUP if the question references earlier results, asks for details about them, comparisons, or clarifications.
- Pick NEW_SEARCH if the user wants a fresh set of papers or switches topics.

User message: "{user_q}"
Previous query: "{prior.last_query or ''}"
Previous paper titles: {titles}

Return JSON: {{"intent": "FOLLOWUP" | "NEW_SEARCH", "reason": "short reason"}}
"""
    out = call_llm_json(prompt, max_new_tokens=256)
    if not isinstance(out, dict):
        return {"intent": "NEW_SEARCH", "reason": "LLM failed"}
    intent = (out.get("intent") or "").upper()
    if intent not in {"FOLLOWUP", "NEW_SEARCH"}:
        intent = "NEW_SEARCH"
    out["intent"] = intent
    return out


def _llm_followup_reply(
    user_q: str,
    paper_ctx: List[Dict[str, Any]],
    original_q: Optional[str] = None,
) -> Optional[str]:
    prompt = (
        "You answer follow-up questions about previously returned papers for a research assistant.\n"
        "Use ONLY the provided papers; do not invent new ones or add external papers.\n"
        "Synthesize a fresh response (do NOT copy provided summaries verbatim) using evidence/snippets.\n"
        "Address the user's request directly; include paper titles or ranks when helpful.\n"
        "Keep it concise: 3-6 sentences, plain text.\n"
        'Return JSON: {"reply": "..."}\n\n'
        f"Original search (for context): {original_q or 'n/a'}\n"
        f"Follow-up request: {user_q}\n"
        f"Papers JSON: {paper_ctx}\n"
    )
    res = call_llm_json_last(prompt, max_new_tokens=512)
    if isinstance(res, dict):
        reply = res.get("reply") or res.get("why")
        if reply:
            return str(reply).strip()
    if isinstance(res, str) and res.strip():
        return res.strip()
    return None


def _fallback_followup_reply(paper_ctx: List[Dict[str, Any]], user_q: str) -> str:
    if not paper_ctx:
        return f"I couldn't find previous papers to answer “{user_q}”. Please run a new search."
    lines = ["Here’s more using the earlier results:"]
    for p in paper_ctx:
        bits = []
        if p.get("title"):
            bits.append(f"{p.get('title')} ({p.get('year') or 'year n/a'})")
        if p.get("authors"):
            bits.append(", ".join(p["authors"]))
        if p.get("summary"):
            bits.append(p["summary"])
        elif p.get("evidence_long"):
            bits.append(p["evidence_long"][0])
        elif p.get("evidence"):
            bits.append(p["evidence"][0])
        line = " — ".join(bits) if bits else f"Paper {p.get('rank')}"
        lines.append(f"- {line}")
    return "\n".join(lines)


async def _maybe_handle_followup(req: "SearchRequest") -> Optional[SearchEnvelope]:
    prior = req.prior_results
    if not prior or not prior.last_papers:
        return None

    # if the user is asking for papers by an author, treat as a fresh search
    if _looks_like_author_query(req.query):
        return None

    heuristic_followup = _looks_like_followup(req.query)
    intent_meta = {"intent": "NEW_SEARCH", "reason": "default"}

    if heuristic_followup:
        intent_meta = {"intent": "FOLLOWUP", "reason": "heuristic match"}
    else:
        intent_meta = await call_maybe_async(_llm_route_followup, req.query, prior, prior.last_papers)

    if (intent_meta.get("intent") or "").upper() != "FOLLOWUP":
        return None

    idxs = _select_paper_indices(req.query, prior.last_papers)
    chosen = [prior.last_papers[i] for i in idxs if i < len(prior.last_papers)]
    paper_ctx = _paper_context_for_prompt(prior.last_papers, idxs, followup_q=req.query, original_q=prior.last_query)
    reply = await call_maybe_async(_llm_followup_reply, req.query, paper_ctx, prior.last_query)
    if not reply:
        reply = _fallback_followup_reply(paper_ctx, req.query)

    context_text = "\n\n---\n\n".join(
        [c.get("summary") or "" for c in paper_ctx if c.get("summary")] or ["Used previous papers for follow-up."]
    )

    sr = SearchResponse(
        query=prior.last_query or req.query,
        refined_query=prior.last_refined_query or prior.last_query or req.query,
        top_score=None,
        hits=[],
        papers=chosen,
        context=context_text,
    )

    analysis_payload = {
        "intent": "FOLLOWUP",
        "router": intent_meta,
        "source_analysis": prior.last_analysis,
        "selected_indices": idxs,
        "paper_ids": [p.paper_id for p in chosen if p.paper_id],
    }

    return SearchEnvelope(
        analysis=analysis_payload,
        results=sr,
        model=MODEL_NAME,
        reply_text=reply,
    )


async def _run_search_impl(req: "SearchRequest") -> SearchEnvelope:
    """
    Core search logic, extracted so it can be wrapped in a cancellable Task.
    This is basically your old /api/search body.
    """
    followup = await _maybe_handle_followup(req)
    if followup:
        return followup

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
