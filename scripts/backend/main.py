# backend/main.py
from fastapi import FastAPI
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

class SearchRequest(BaseModel):
    q: str
    k: int = Field(5, ge=1, le=100) 
    show_scores: bool = True

@app.post("/api/search")
def search(req: SearchRequest):
    return query_rag(req.q, k=req.k, show_scores=req.show_scores)
