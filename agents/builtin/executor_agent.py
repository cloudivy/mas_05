"""Executor Agent — implements the plan."""
from agents.base_agent import BaseAgent


class ExecutorAgent(BaseAgent):
    agent_id = "executor"
    name = "Executor"
    role = "Implements the plan and produces deliverables"
    icon = "▣"
    color = "#8b5cf6"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Executor agent. Take the plan and produce concrete output — "
            "code, prose, decisions, or structured data. Be specific and actionable. Max 200 words."
        )
