"""
RAG Store — lightweight document store with keyword retrieval.

Documents are persisted in rag/.store/{collection}.json.

Upgrade path: replace _score() with sentence-transformers embeddings
or plug in ChromaDB / Pinecone for production semantic search.
"""
from __future__ import annotations
import json
import re
import time
import uuid
from pathlib import Path

RAG_DIR = Path(__file__).parent / ".store"


class RAGStore:
    def __init__(self, collection: str = "default"):
        self.collection = collection
        self.path = RAG_DIR / f"{collection}.json"
        RAG_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def _write(self, docs: list[dict]):
        self.path.write_text(json.dumps(docs, indent=2))

    def _score(self, query: str, text: str) -> float:
        """Jaccard keyword overlap. Replace with embedding similarity for production."""
        q = set(re.findall(r"\w+", query.lower()))
        d = set(re.findall(r"\w+", text.lower()))
        if not d:
            return 0.0
        return len(q & d) / len(q | d)

    # ── Public API ───────────────────────────────────────────────────────────
    def add_document(self, title: str, content: str, metadata: dict | None = None) -> str:
        docs = self._read()
        doc_id = uuid.uuid4().hex[:8]
        docs.append({
            "id": doc_id,
            "title": title,
            "content": content,
            "metadata": metadata or {},
            "added_at": time.time(),
        })
        self._write(docs)
        return doc_id

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        docs = self._read()
        scored = [
            (self._score(query, d["title"] + " " + d["content"]), d)
            for d in docs
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [d for score, d in scored[:top_k] if score > 0]

    def get_context_string(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        return "\n\n".join(
            f"[RAG: {r['title']}]\n{r['content'][:300]}" for r in results
        )

    def list_documents(self) -> list[dict]:
        return [
            {"id": d["id"], "title": d["title"], "added_at": d["added_at"]}
            for d in self._read()
        ]

    def delete_document(self, doc_id: str) -> bool:
        docs = self._read()
        new_docs = [d for d in docs if d["id"] != doc_id]
        if len(new_docs) == len(docs):
            return False
        self._write(new_docs)
        return True

    def clear(self):
        self._write([])
