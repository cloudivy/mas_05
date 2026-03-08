"""
Tool Registry — register and invoke callable tools per agent.

Built-in tools: calculator, timestamp, web_search_mock
Add your own by calling registry.register(...) in an agent's _register_tools().
"""
from __future__ import annotations
import math
import time
from typing import Any, Callable


class Tool:
    def __init__(self, name: str, fn: Callable, description: str, params: dict):
        self.name = name
        self.fn = fn
        self.description = description
        self.params = params

    def invoke(self, **kwargs) -> Any:
        return self.fn(**kwargs)

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.params}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._register_builtins()

    def _register_builtins(self):
        self.register(
            name="calculator",
            fn=lambda expression: str(
                eval(expression, {"__builtins__": {}}, {"math": math})
            ),
            description="Evaluate a safe math expression",
            params={"expression": {"type": "string", "description": "Math expression"}},
        )
        self.register(
            name="timestamp",
            fn=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            description="Return current UTC timestamp",
            params={},
        )
        self.register(
            name="web_search_mock",
            fn=lambda query: f"[Mock results for '{query}'] — connect a real search API here.",
            description="Mock web search (replace with SerpAPI / Tavily)",
            params={"query": {"type": "string", "description": "Search query"}},
        )

    def register(self, name: str, fn: Callable, description: str, params: dict):
        self._tools[name] = Tool(name=name, fn=fn, description=description, params=params)

    def invoke(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not registered")
        return self._tools[name].invoke(**kwargs)

    def list_tools(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    def schemas(self) -> list[dict]:
        return self.list_tools()
