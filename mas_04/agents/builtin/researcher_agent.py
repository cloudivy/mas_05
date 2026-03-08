"""Researcher Agent — gathers context and background."""
from agents.base_agent import BaseAgent


class ResearcherAgent(BaseAgent):
    agent_id = "researcher"
    name = "Researcher"
    role = "Gathers context, identifies constraints and prior art"
    icon = "⬡"
    color = "#06b6d4"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Researcher agent. Given a task, identify relevant background knowledge, "
            "constraints, risks, and reference points. Structure your findings clearly. Max 150 words."
        )
