"""
Base Agent — all agents in the framework extend this class.

Supports both OpenAI (gpt-4o-mini / gpt-4o) and Anthropic (claude-*) backends.
Set provider via api_key prefix:
  - sk-ant-...  → Anthropic
  - sk-...      → OpenAI
Or set provider="anthropic" | "openai" explicitly.

To add a new agent: subclass BaseAgent, implement `system_prompt`,
optionally override `tools` and `_register_tools`.
"""
from __future__ import annotations
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from memory.memory_store import MemoryStore
from tools.tool_registry import ToolRegistry
from eval.agent_metrics import AgentMetrics


@dataclass
class AgentMessage:
    role: str           # "user" | "assistant" | "system"
    content: str
    agent_id: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    agent_id: str
    agent_name: str
    output: str
    messages: list[AgentMessage]
    metrics: dict
    tool_calls: list[dict]
    elapsed_s: float
    success: bool
    mode: str = "api"
    error: Optional[str] = None


def _detect_provider(api_key: str) -> str:
    """Detect API provider from key prefix."""
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    return "openai"


class BaseAgent(ABC):
    """
    Extend this to create a new agent.
    Minimum required: define `agent_id`, `name`, `role`, and `system_prompt`.
    """

    agent_id: str = "base"
    name: str = "Base Agent"
    role: str = "Generic agent"
    icon: str = "◎"
    color: str = "#888888"

    def __init__(
        self,
        api_key: str = "",
        simulate: bool = False,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or ""
        self.simulate = simulate
        self.provider = provider or (_detect_provider(self.api_key) if self.api_key else "openai")
        self.model = model  # None → use default per provider
        self.memory = MemoryStore(namespace=self.agent_id)
        self.tool_registry = ToolRegistry()
        self.metrics = AgentMetrics(agent_id=self.agent_id)
        self._register_tools()

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Define this agent's persona and instructions."""
        ...

    def _register_tools(self):
        """Override to register agent-specific tools."""
        pass

    def _simulated_response(self, task: str, context: str) -> str:
        return (
            f"[SIMULATED] {self.name} processed: '{task[:80]}'\n\n"
            f"Role: {self.role}\n"
            f"Context tokens received: {len(context.split())}\n"
            f"Output: Placeholder — provide an API key for real LLM responses."
        )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    def _call_openai(self, task: str, context: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        model = self.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        user_content = f"Task: {task}"
        if context.strip():
            user_content += f"\n\nContext from prior agents:\n{context}"

        response = client.chat.completions.create(
            model=model,
            max_tokens=600,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": user_content},
            ],
        )
        return response.choices[0].message.content

    # ── Anthropic ─────────────────────────────────────────────────────────────
    def _call_anthropic(self, task: str, context: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        model = self.model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

        user_content = f"Task: {task}"
        if context.strip():
            user_content += f"\n\nContext from prior agents:\n{context}"

        response = client.messages.create(
            model=model,
            max_tokens=600,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return response.content[0].text

    # ── Dispatch ──────────────────────────────────────────────────────────────
    def _call_api(self, task: str, context: str) -> str:
        if self.provider == "anthropic":
            return self._call_anthropic(task, context)
        return self._call_openai(task, context)

    # ── Main entry ────────────────────────────────────────────────────────────
    def run(self, task: str, context: str = "") -> AgentResult:
        """Execute this agent. Called by the pipeline."""
        start = time.time()
        messages: list[AgentMessage] = []
        tool_calls: list[dict] = []
        error = None
        mode = "simulated"

        # Inject relevant memories
        memories = self.memory.retrieve(query=task, top_k=3)
        if memories:
            context += "\n\n[Memory]\n" + "\n".join(f"- {m}" for m in memories)

        try:
            if self.simulate or not self.api_key:
                output = self._simulated_response(task, context)
                mode = "simulated"
            else:
                try:
                    output = self._call_api(task, context)
                    mode = self.provider
                except Exception as e:
                    output = self._simulated_response(task, context)
                    output += f"\n\n[API fallback: {e}]"
                    mode = "fallback"
                    error = str(e)

            # Persist output in memory
            self.memory.store(
                key=f"run_{uuid.uuid4().hex[:8]}",
                value=output,
                tags=[task[:40]],
            )

            elapsed = time.time() - start
            metrics = self.metrics.record_run(
                task=task,
                output=output,
                context=context,
                elapsed=elapsed,
                mode=mode,
            )

            messages.append(AgentMessage(role="assistant", content=output, agent_id=self.agent_id))
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                output=output,
                messages=messages,
                metrics=metrics,
                tool_calls=tool_calls,
                elapsed_s=elapsed,
                success=True,
                mode=mode,
                error=error,
            )

        except Exception as e:
            elapsed = time.time() - start
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                output="",
                messages=messages,
                metrics={},
                tool_calls=tool_calls,
                elapsed_s=elapsed,
                success=False,
                mode="error",
                error=str(e),
            )
