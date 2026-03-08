"""
Orchestrator — runs the multi-agent pipeline.

Mode 1 (Sequential): agents run in fixed order, each receiving prior context.
Mode 2 (Hierarchical): OrchestratorAgent dynamically selects next agent. [roadmap]

This module is the core runtime called by the FastAPI layer.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.registry import get_agent, DEFAULT_PIPELINE, list_agents
from agents.base_agent import AgentResult
from eval.agent_metrics import AgentMetrics


@dataclass
class PipelineRun:
    run_id: str
    task: str
    mode: str                         # "sequential" | "hierarchical"
    pipeline: list[str]               # agent_ids in execution order
    results: list[AgentResult] = field(default_factory=list)
    status: str = "pending"           # pending | running | complete | error
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "mode": self.mode,
            "pipeline": self.pipeline,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": round(
                (self.finished_at or time.time()) - self.started_at, 2
            ),
            "results": [
                {
                    "agent_id":   r.agent_id,
                    "agent_name": r.agent_name,
                    "output":     r.output,
                    "metrics":    r.metrics,
                    "elapsed_s":  round(r.elapsed_s, 3),
                    "success":    r.success,
                    "mode":       r.mode,
                    "error":      r.error,
                }
                for r in self.results
            ],
        }


def run_pipeline(
    task: str,
    api_key: str = "",
    pipeline: list[str] | None = None,
    mode: str = "sequential",
    simulate: bool = False,
    provider: str | None = None,
    model: str | None = None,
    on_agent_start: Optional[Callable[[str], None]] = None,
    on_agent_done: Optional[Callable[[str, AgentResult], None]] = None,
) -> PipelineRun:
    """
    Run the full multi-agent pipeline.

    Args:
        task          : The user's task description
        api_key       : OpenAI or Anthropic API key (empty → simulate)
        pipeline      : List of agent_ids to run (default: DEFAULT_PIPELINE)
        mode          : "sequential" | "hierarchical"
        simulate      : Force simulation even if API key is set
        provider      : "openai" | "anthropic" (auto-detected from key if omitted)
        model         : Override default model per provider
        on_agent_start: Callback(agent_id) fired before each agent
        on_agent_done : Callback(agent_id, result) fired after each agent
    """
    pipeline = pipeline or DEFAULT_PIPELINE
    run = PipelineRun(
        run_id=uuid.uuid4().hex[:12],
        task=task,
        mode=mode,
        pipeline=pipeline,
        status="running",
    )

    context = ""

    for agent_id in pipeline:
        if on_agent_start:
            on_agent_start(agent_id)

        agent = get_agent(
            agent_id,
            api_key=api_key,
            simulate=simulate,
            provider=provider,
            model=model,
        )
        result = agent.run(task=task, context=context)
        run.results.append(result)

        if result.success:
            context += f"\n\n[{result.agent_name}]\n{result.output}"
        else:
            run.status = "error"
            run.error = f"{agent_id}: {result.error}"
            run.finished_at = time.time()
            return run

        if on_agent_done:
            on_agent_done(agent_id, result)

    run.status = "complete"
    run.finished_at = time.time()
    return run


def get_system_status(api_key: str = "") -> dict:
    """Health check — lists all agents and their latest ASI."""
    agents = list_agents()
    for a in agents:
        m = AgentMetrics(agent_id=a["id"])
        a["latest_asi"] = m.latest_asi()
        a["run_count"] = len(m.get_history(100))
    return {
        "agents": agents,
        "default_pipeline": DEFAULT_PIPELINE,
        "api_configured": bool(api_key),
    }
