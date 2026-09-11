import time
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.generate import generate_answer

app = FastAPI(title="BMW Intelligence RAG API")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    os.getenv("FRONTEND_URL", "*"),  #Allows Vercel domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    manufacturer: str = "BMW"
    model: str = "735i M Sport"
    model_year: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    history: List[Message] = []

class LatencyBreakdown(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float

class DocumentSource(BaseModel):
    doc_id: int
    title: str
    year: str
    url: str
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    manufacturer: str
    model: str
    model_year: Optional[str]
    latency: LatencyBreakdown
    sources: List[DocumentSource]

@app.post("/api/v1/ask", response_model=QueryResponse)
async def ask_documentation(request: QueryRequest):
    try:
        history_dicts = [msg.model_dump() for msg in request.history]
        t_start = time.perf_counter()

        answer, effective_year, sources, t_retrieval_end, t_gen_end = generate_answer(
            query_text=request.query,
            manufacturer=request.manufacturer,
            model=request.model,
            model_year=request.model_year,
            top_k=request.top_k,
            history=history_dicts,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        retrieval_ms = t_retrieval_end * 1000
        generation_ms = t_gen_end * 1000

        return QueryResponse(
            answer=answer,
            manufacturer=request.manufacturer,
            model=request.model,
            model_year=effective_year,
            latency=LatencyBreakdown(
                retrieval_ms=round(retrieval_ms, 2),
                generation_ms=round(generation_ms, 2),
                total_ms=round(total_ms, 2),
            ),
            sources=sources,
        )
    except Exception as e:
        print(f"Internal API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))