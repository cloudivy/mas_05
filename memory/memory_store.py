"""
Memory Store — lightweight per-agent persistent memory backed by JSON.

Each agent gets its own namespace file at memory/.store/{namespace}.json.

Upgrade path: swap the JSON backend for Redis / ChromaDB / Postgres
by subclassing MemoryStore and overriding store / retrieve.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Optional

MEMORY_DIR = Path(__file__).parent / ".store"


class MemoryStore:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.path = MEMORY_DIR / f"{namespace}.json"
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    # ── I/O ──────────────────────────────────────────────────────────────────
    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {}

    def _write(self, data: dict):
        self.path.write_text(json.dumps(data, indent=2))

    # ── Public API ───────────────────────────────────────────────────────────
    def store(self, key: str, value: str, tags: list[str] | None = None):
        """Persist a memory entry."""
        data = self._read()
        data[key] = {"value": value, "tags": tags or [], "ts": time.time()}
        self._write(data)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """Return top-k entries most relevant to the query (keyword overlap)."""
        data = self._read()
        if not data:
            return []
        query_words = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[int, str]] = []
        for entry in data.values():
            text = entry["value"] + " " + " ".join(entry.get("tags", []))
            words = set(re.findall(r"\w+", text.lower()))
            score = len(query_words & words)
            if score > 0:
                scored.append((score, entry["value"]))
        scored.sort(reverse=True)
        return [v for _, v in scored[:top_k]]

    def get_all(self) -> dict:
        """Dump all entries for this namespace."""
        return self._read()

    def clear(self):
        """Wipe the namespace."""
        self._write({})

    def summary(self) -> dict:
        data = self._read()
        return {"namespace": self.namespace, "entries": len(data)}
