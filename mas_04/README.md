# 🤖 MAS-04 · Multi-Agent Framework

A pluggable, production-ready LLM multi-agent system with:

- **Agent Registry** — drop in new agents with one file
- **Sequential → Hierarchical** routing
- **Persistent Memory** per agent (JSON → swap for Redis)
- **RAG** document store with keyword retrieval
- **Tool Registry** per agent
- **Agent Stability Index (ASI)** — drift evaluation & observability
- **FastAPI backend** + standalone **frontend** (GitHub Pages ready)
- **OpenAI + Anthropic** support — switch with your API key prefix

---

## 🗂 Repo Structure

```
mas_04/
├── agents/
│   ├── base_agent.py              # BaseAgent — extend this for new agents
│   ├── registry.py                # Agent registry + DEFAULT_PIPELINE
│   └── builtin/
│       ├── orchestrator_agent.py  # Decomposes tasks
│       ├── researcher_agent.py    # Gathers context
│       ├── planner_agent.py       # Creates execution roadmap
│       ├── executor_agent.py      # Produces deliverables
│       └── critic_agent.py        # Reviews quality
├── memory/
│   └── memory_store.py            # Per-agent JSON memory (swap for Redis)
├── rag/
│   └── rag_store.py               # Document store + keyword retrieval
├── tools/
│   └── tool_registry.py           # Pluggable tools per agent
├── orchestrator/
│   └── pipeline.py                # Sequential pipeline runner
├── eval/
│   └── agent_metrics.py           # ASI scoring + run history
├── api/
│   └── main.py                    # FastAPI backend (all routes)
├── frontend/
│   └── index.html                 # Standalone React UI
├── tests/
│   └── test_core.py               # pytest test suite
├── .github/workflows/ci.yml       # CI + GitHub Pages auto-deploy
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/mas_04
cd mas_04
pip install -r requirements.txt
cp .env.example .env   # add your API key
```

### 2. Configure your API key

Edit `.env`:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# OR Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

The framework auto-detects provider from the key prefix (`sk-ant-` → Anthropic, `sk-` → OpenAI).

### 3. Run the backend

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Open the frontend

```bash
open frontend/index.html
# or visit http://localhost:8000/ui
```

### 5. Demo / no backend

Open `frontend/index.html` directly in a browser — enter your API key in the UI and run the pipeline entirely from the browser with no backend required.

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/agents` | List all registered agents |
| POST | `/run` | Run the full pipeline |
| GET | `/run/{run_id}` | Get a past run |
| GET | `/runs` | List all runs (in-memory) |
| GET | `/metrics` | Cross-agent ASI summary |
| GET | `/metrics/{agent_id}` | Per-agent ASI history |
| GET | `/status` | System status + ASI per agent |
| POST | `/rag/add` | Add a RAG document |
| GET | `/rag/list` | List RAG documents |
| DELETE | `/rag/{doc_id}` | Delete a RAG document |
| GET | `/memory/{agent_id}` | Inspect agent memory |
| DELETE | `/memory/{agent_id}` | Clear agent memory |

### Run the pipeline via API

```bash
# OpenAI
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Design a recommendation engine", "api_key": "sk-...", "model": "gpt-4o-mini"}'

# Anthropic
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze context drift in LLM agents", "api_key": "sk-ant-...", "provider": "anthropic", "model": "claude-haiku-4-5-20251001"}'

# Simulate (no API key)
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"task": "Build a REST API", "simulate": true}'
```

---

## 📊 Agent Stability Index (ASI)

Each agent run is scored across 4 dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Task Adherence | 35% | Task keyword coverage in output |
| Response Consistency | 30% | Output length variance over last 5 runs |
| Reasoning Stability | 25% | Keyword overlap with prior run output |
| Latency Stability | 10% | Elapsed time variance over last 5 runs |

**ASI = weighted mean (0–100).** Persisted to `eval/.store/{agent_id}_metrics.json`.

---

## ➕ Adding a New Agent

1. Create `agents/builtin/my_agent.py`:

```python
from agents.base_agent import BaseAgent

class MyAgent(BaseAgent):
    agent_id = "my_agent"
    name = "My Agent"
    role = "Does something specific"
    icon = "◆"
    color = "#f97316"

    @property
    def system_prompt(self) -> str:
        return "You are MyAgent. Your role is..."
```

2. Register in `agents/registry.py`:

```python
from agents.builtin.my_agent import MyAgent
REGISTRY["my_agent"] = MyAgent
```

3. Optionally add to `DEFAULT_PIPELINE`.

The API and UI pick it up automatically — no other changes needed.

---

## 🌐 GitHub Pages Deployment

The frontend auto-deploys on every push to `main` via GitHub Actions.

1. Go to **Settings → Pages → Source → GitHub Actions**
2. Optionally add `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` to **Settings → Secrets**
3. Push to `main` — live at `https://YOUR_USERNAME.github.io/mas_04/`

The frontend works in **full standalone mode** — enter your API key directly in the UI. No backend required for GitHub Pages.

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests cover: memory roundtrip, RAG retrieval/delete, tool registry, ASI computation, simulated pipeline end-to-end.

---

## 🛣 Roadmap

- [ ] Hierarchical orchestration (dynamic agent selection)
- [ ] Streaming API responses (SSE)
- [ ] Embedding-based RAG (sentence-transformers / ChromaDB)
- [ ] A2A (Agent-to-Agent) direct messaging
- [ ] Persistent run storage (SQLite / PostgreSQL)
- [ ] Multi-run drift report export
- [ ] Docker Compose one-command deployment

---

## 📄 License

MIT
