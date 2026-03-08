"""
FastAPI Backend for MAS-04.

Routes
------
GET  /                      Health check
GET  /agents                List all registered agents
POST /run                   Run the full pipeline (blocking)
GET  /run/{run_id}          Retrieve a past run
GET  /runs                  List all runs (in-memory, current session)
GET  /metrics/{agent_id}    Per-agent ASI history
GET  /metrics               Cross-agent metrics summary
GET  /status                System status + latest ASI per agent
POST /rag/add               Add a RAG document
GET  /rag/list              List RAG documents
DELETE /rag/{doc_id}        Delete a RAG document
GET  /memory/{agent_id}     Inspect agent memory
DELETE /memory/{agent_id}   Clear agent memory

Run with:
    uvicorn api.main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path

# Make project root importable when launched from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from orchestrator.pipeline import run_pipeline, get_system_status, PipelineRun
from agents.registry import list_agents, DEFAULT_PIPELINE
from eval.agent_metrics import AgentMetrics
from rag.rag_store import RAGStore
from memory.memory_store import MemoryStore

app = FastAPI(
    title="MAS-04 API",
    description="Pluggable LLM multi-agent system — memory, RAG, tools, ASI evaluation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run store (swap for SQLite / Postgres for persistence)
_runs: dict[str, dict] = {}


# ── Request models ─────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    task: str
    api_key: str = ""
    pipeline: Optional[list[str]] = None
    simulate: bool = False
    provider: Optional[str] = None   # "openai" | "anthropic"
    model: Optional[str] = None      # override default model

class RAGAddRequest(BaseModel):
    title: str
    content: str
    collection: str = "default"
    metadata: dict = {}


# ── Core routes ────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "mas-04", "version": "1.0.0"}


@app.get("/agents")
def get_agents():
    return {"agents": list_agents(), "default_pipeline": DEFAULT_PIPELINE}


@app.post("/run")
def run(req: RunRequest):
    if not req.task.strip():
        raise HTTPException(400, "task cannot be empty")

    # Fallback to env vars if no key provided in request
    api_key = req.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""

    run_result: PipelineRun = run_pipeline(
        task=req.task,
        api_key=api_key,
        pipeline=req.pipeline,
        simulate=req.simulate or not api_key,
        provider=req.provider,
        model=req.model,
    )
    run_dict = run_result.to_dict()
    _runs[run_dict["run_id"]] = run_dict
    return run_dict


@app.get("/run/{run_id}")
def get_run(run_id: str):
    if run_id not in _runs:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return _runs[run_id]


@app.get("/runs")
def list_runs():
    return {"runs": list(_runs.values())}


@app.get("/metrics/{agent_id}")
def get_agent_metrics(agent_id: str, last_n: int = Query(20, ge=1, le=100)):
    m = AgentMetrics(agent_id=agent_id)
    history = m.get_history(last_n)
    return {"agent_id": agent_id, "history": history, "run_count": len(history)}


@app.get("/metrics")
def get_all_metrics():
    agents = list_agents()
    summary = []
    for a in agents:
        m = AgentMetrics(agent_id=a["id"])
        history = m.get_history(20)
        asi_scores = [h["scores"].get("asi", 0) for h in history if h.get("scores")]
        summary.append({
            "agent_id":   a["id"],
            "name":       a["name"],
            "color":      a["color"],
            "run_count":  len(history),
            "latest_asi": asi_scores[-1] if asi_scores else None,
            "avg_asi":    round(sum(asi_scores) / len(asi_scores), 1) if asi_scores else None,
            "history":    history[-10:],
        })
    return {"agents": summary}


@app.get("/status")
def status(api_key: str = ""):
    return get_system_status(api_key=api_key)


# ── RAG ────────────────────────────────────────────────────────────────────────

@app.post("/rag/add")
def rag_add(req: RAGAddRequest):
    store = RAGStore(collection=req.collection)
    doc_id = store.add_document(title=req.title, content=req.content, metadata=req.metadata)
    return {"doc_id": doc_id, "collection": req.collection}


@app.get("/rag/list")
def rag_list(collection: str = "default"):
    store = RAGStore(collection=collection)
    return {"collection": collection, "documents": store.list_documents()}


@app.delete("/rag/{doc_id}")
def rag_delete(doc_id: str, collection: str = "default"):
    store = RAGStore(collection=collection)
    if not store.delete_document(doc_id):
        raise HTTPException(404, f"Document '{doc_id}' not found")
    return {"deleted": doc_id}


# ── Memory ─────────────────────────────────────────────────────────────────────

@app.get("/memory/{agent_id}")
def get_memory(agent_id: str):
    mem = MemoryStore(namespace=agent_id)
    return {"agent_id": agent_id, "memories": mem.get_all()}


@app.delete("/memory/{agent_id}")
def clear_memory(agent_id: str):
    mem = MemoryStore(namespace=agent_id)
    mem.clear()
    return {"cleared": agent_id}


# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

    @app.get("/ui")
    def serve_ui():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
