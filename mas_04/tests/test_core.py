"""
Core test suite for MAS-04.
Run: pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from memory.memory_store import MemoryStore
from rag.rag_store import RAGStore
from tools.tool_registry import ToolRegistry
from eval.agent_metrics import AgentMetrics


# ── Memory ────────────────────────────────────────────────────────────────────

def test_memory_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.memory_store.MEMORY_DIR", tmp_path)
    m = MemoryStore(namespace="test")
    m.store("k1", "FastAPI Python web framework async", tags=["python"])
    results = m.retrieve("Python async web", top_k=1)
    assert len(results) == 1
    assert "FastAPI" in results[0]


def test_memory_retrieve_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.memory_store.MEMORY_DIR", tmp_path)
    m = MemoryStore(namespace="empty_ns")
    results = m.retrieve("anything", top_k=3)
    assert results == []


def test_memory_clear(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.memory_store.MEMORY_DIR", tmp_path)
    m = MemoryStore(namespace="clear_test")
    m.store("k1", "some value")
    m.clear()
    assert m.get_all() == {}


# ── RAG ───────────────────────────────────────────────────────────────────────

def test_rag_add_retrieve(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.rag_store.RAG_DIR", tmp_path)
    store = RAGStore(collection="test")
    store.add_document("FastAPI Guide", "FastAPI is a modern Python web framework for building APIs")
    results = store.retrieve("Python web API")
    assert len(results) >= 1
    assert results[0]["title"] == "FastAPI Guide"


def test_rag_delete(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.rag_store.RAG_DIR", tmp_path)
    store = RAGStore(collection="del_test")
    doc_id = store.add_document("To Delete", "Some content")
    assert store.delete_document(doc_id) is True
    assert store.delete_document("nonexistent") is False


def test_rag_context_string(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.rag_store.RAG_DIR", tmp_path)
    store = RAGStore(collection="ctx_test")
    store.add_document("Agent Systems", "Multi-agent systems coordinate tasks across specialist agents")
    ctx = store.get_context_string("multi-agent tasks")
    assert "[RAG:" in ctx


# ── Tools ─────────────────────────────────────────────────────────────────────

def test_tool_registry_builtins():
    reg = ToolRegistry()
    names = [t["name"] for t in reg.list_tools()]
    assert "calculator" in names
    assert "timestamp" in names
    assert "web_search_mock" in names


def test_calculator_tool():
    reg = ToolRegistry()
    result = reg.invoke("calculator", expression="2 ** 10")
    assert result == "1024"


def test_calculator_math_import():
    reg = ToolRegistry()
    result = reg.invoke("calculator", expression="math.sqrt(144)")
    assert result == "12.0"


def test_tool_not_found():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="not registered"):
        reg.invoke("nonexistent_tool")


# ── ASI Metrics ───────────────────────────────────────────────────────────────

def test_asi_scores_range(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.agent_metrics.EVAL_DIR", tmp_path)
    m = AgentMetrics(agent_id="test_agent")
    scores = m.record_run(
        task="build a recommendation engine for content platform",
        output="A recommendation engine uses collaborative filtering and content-based methods to suggest items.",
        context="",
        elapsed=1.2,
        mode="simulated",
    )
    assert "asi" in scores
    assert 0 <= scores["asi"] <= 100
    assert all(0 <= scores[k] <= 100 for k in ["response_consistency","reasoning_stability","task_adherence","latency_stability"])


def test_asi_history(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.agent_metrics.EVAL_DIR", tmp_path)
    m = AgentMetrics(agent_id="hist_agent")
    for i in range(5):
        m.record_run(task=f"task {i}", output=f"output for task {i} with relevant keywords", context="", elapsed=0.5+i*0.1, mode="sim")
    history = m.get_history(10)
    assert len(history) == 5
    assert all("scores" in h for h in history)


def test_asi_task_adherence_high(tmp_path, monkeypatch):
    """Output that mirrors task keywords should have high task adherence."""
    monkeypatch.setattr("eval.agent_metrics.EVAL_DIR", tmp_path)
    m = AgentMetrics(agent_id="ta_agent")
    task = "analyze machine learning model performance drift detection"
    output = "Analyzing machine learning model performance drift detection using statistical methods and monitoring."
    scores = m.record_run(task=task, output=output, context="", elapsed=1.0, mode="sim")
    assert scores["task_adherence"] >= 60


def test_asi_latest(tmp_path, monkeypatch):
    monkeypatch.setattr("eval.agent_metrics.EVAL_DIR", tmp_path)
    m = AgentMetrics(agent_id="latest_agent")
    assert m.latest_asi() is None
    m.record_run(task="test task", output="test output for task", context="", elapsed=0.5, mode="sim")
    assert m.latest_asi() is not None


# ── Pipeline (simulate=True) ──────────────────────────────────────────────────

def test_pipeline_simulate(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.memory_store.MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr("rag.rag_store.RAG_DIR", tmp_path / "rag")
    monkeypatch.setattr("eval.agent_metrics.EVAL_DIR", tmp_path / "eval")

    from orchestrator.pipeline import run_pipeline
    result = run_pipeline(task="Test task for simulation", simulate=True)

    assert result.status == "complete"
    assert len(result.results) == 5
    assert all(r.success for r in result.results)


def test_pipeline_to_dict(tmp_path, monkeypatch):
    monkeypatch.setattr("memory.memory_store.MEMORY_DIR", tmp_path / "mem")
    monkeypatch.setattr("rag.rag_store.RAG_DIR", tmp_path / "rag")
    monkeypatch.setattr("eval.agent_metrics.EVAL_DIR", tmp_path / "eval")

    from orchestrator.pipeline import run_pipeline
    result = run_pipeline(task="Test serialization", simulate=True)
    d = result.to_dict()

    assert "run_id" in d
    assert "results" in d
    assert d["status"] == "complete"
    assert d["elapsed_s"] > 0
