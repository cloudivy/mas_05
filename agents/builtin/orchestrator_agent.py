"""Orchestrator Agent — decomposes tasks and delegates."""
from agents.base_agent import BaseAgent


class OrchestratorAgent(BaseAgent):
    agent_id = "orchestrator"
    name = "Orchestrator"
    role = "Decomposes tasks and delegates to specialist agents"
    icon = "◈"
    color = "#f59e0b"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Orchestrator in a multi-agent AI system. "
            "Your job is to receive a task, break it into clear subtasks, "
            "and describe which specialist agent should handle each part. "
            "Be concise, structured, and precise. Max 150 words."
        )
