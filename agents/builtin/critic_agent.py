"""Critic Agent — reviews and scores output."""
from agents.base_agent import BaseAgent


class CriticAgent(BaseAgent):
    agent_id = "critic"
    name = "Critic"
    role = "Reviews output quality and flags gaps"
    icon = "◬"
    color = "#ef4444"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Critic agent. Review the executor's output against the original task. "
            "Score it 0-100, list strengths, gaps, and concrete improvements. Max 150 words."
        )
