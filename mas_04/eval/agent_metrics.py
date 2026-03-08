"""
Agent Metrics & Evaluation — per-agent run statistics and ASI scoring.

Agent Stability Index (ASI) — 4 dimensions:
  1. Response Consistency  (0.30) — output length variance over last 5 runs
  2. Reasoning Stability   (0.25) — keyword overlap with prior run output
  3. Task Adherence        (0.35) — task keyword coverage in agent output
  4. Latency Stability     (0.10) — elapsed time variance over last 5 runs

ASI = weighted mean (0–100). Persisted to eval/.store/{agent_id}_metrics.json.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from typing import Optional

EVAL_DIR = Path(__file__).parent / ".store"


class AgentMetrics:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.path = EVAL_DIR / f"{agent_id}_metrics.json"
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"runs": []})

    def _read(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return {"runs": []}

    def _write(self, data: dict):
        self.path.write_text(json.dumps(data, indent=2))

    def _keyword_overlap(self, a: str, b: str) -> float:
        wa = set(re.findall(r"\w+", a.lower()))
        wb = set(re.findall(r"\w+", b.lower()))
        if not wa or not wb:
            return 1.0
        return len(wa & wb) / len(wa | wb)

    def _compute_asi(self, run: dict, prior_runs: list[dict]) -> dict:
        scores: dict[str, float] = {}

        # 1. Response Consistency — output length variance
        lengths = [
            len(r["output"].split()) for r in prior_runs[-5:]
        ] + [len(run["output"].split())]
        if len(lengths) > 1:
            mean_l = sum(lengths) / len(lengths)
            variance = sum((l - mean_l) ** 2 for l in lengths) / len(lengths)
            scores["response_consistency"] = max(
                0.0, 100 - min(variance / max(mean_l, 1) * 100, 100)
            )
        else:
            scores["response_consistency"] = 100.0

        # 2. Reasoning Stability — keyword overlap with last run
        if prior_runs:
            overlap = self._keyword_overlap(run["output"], prior_runs[-1]["output"])
            scores["reasoning_stability"] = round(overlap * 100, 1)
        else:
            scores["reasoning_stability"] = 100.0

        # 3. Task Adherence — task keywords found in output
        task_words = set(re.findall(r"\w{4,}", run["task"].lower()))
        out_words = set(re.findall(r"\w+", run["output"].lower()))
        if task_words:
            scores["task_adherence"] = round(
                len(task_words & out_words) / len(task_words) * 100, 1
            )
        else:
            scores["task_adherence"] = 100.0

        # 4. Latency Stability — elapsed time variance
        latencies = [r["elapsed_s"] for r in prior_runs[-5:]] + [run["elapsed_s"]]
        if len(latencies) > 1:
            mean_lat = sum(latencies) / len(latencies)
            lat_var = sum((l - mean_lat) ** 2 for l in latencies) / len(latencies)
            scores["latency_stability"] = max(0.0, 100 - min(lat_var * 10, 100))
        else:
            scores["latency_stability"] = 100.0

        # ASI composite
        weights = {
            "response_consistency": 0.30,
            "reasoning_stability":  0.25,
            "task_adherence":       0.35,
            "latency_stability":    0.10,
        }
        asi = sum(scores[k] * w for k, w in weights.items())
        scores["asi"] = round(asi, 1)
        return scores

    def record_run(
        self,
        task: str,
        output: str,
        context: str,
        elapsed: float,
        mode: str,
    ) -> dict:
        data = self._read()
        run = {
            "task": task[:200],
            "output": output,
            "elapsed_s": round(elapsed, 3),
            "mode": mode,
            "ts": time.time(),
        }
        scores = self._compute_asi(run, data["runs"])
        run["scores"] = scores
        data["runs"].append(run)
        self._write(data)
        return scores

    def get_history(self, last_n: int = 20) -> list[dict]:
        runs = self._read().get("runs", [])[-last_n:]
        return [
            {
                "task": r["task"],
                "mode": r["mode"],
                "elapsed_s": r["elapsed_s"],
                "ts": r["ts"],
                "scores": r.get("scores", {}),
            }
            for r in runs
        ]

    def latest_asi(self) -> Optional[float]:
        history = self.get_history(1)
        if history:
            return history[-1]["scores"].get("asi")
        return None
