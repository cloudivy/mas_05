"""Planner Agent — structures the execution approach."""
from agents.base_agent import BaseAgent


class PlannerAgent(BaseAgent):
    agent_id = "planner"
    name = "Planner"
    role = "Creates a structured execution roadmap"
    icon = "◎"
    color = "#10b981"

    @property
    def system_prompt(self) -> str:
        return (
            "You are the Planner agent. Based on the task and research context, create a clear "
            "phased execution plan with steps, dependencies, and estimates. Max 150 words."
        )
