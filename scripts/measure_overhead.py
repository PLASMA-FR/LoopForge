#!/usr/bin/env python3
"""Measure orchestration with a scripted backend. No inference speed claims."""
import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopforge.backend import Completion
from loopforge.config import Config
from loopforge.engine import Engine
from loopforge.state import Store


class Scripted:
    def __init__(self):
        self.turn = 0

    async def complete(self, messages, **kwargs):
        self.turn += 1
        obj = {"tool": "list", "args": {}} if self.turn % 2 else {"tool": "finish", "args": {"summary": "Inspected"}}
        return Completion({"role": "assistant", "content": json.dumps(obj)},
                          {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, 0, "stop")


async def measure(runs):
    times = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        project = root / "project"
        project.mkdir()
        (project / "file.py").write_text("x = 1\n")
        cfg = Config(workspace=str(project), state_dir=str(root / "state")).validate()
        store = Store(cfg.state_dir)
        try:
            for _ in range(runs):
                start = time.perf_counter()
                result = await Engine(cfg, Scripted(), store).run("Inspect repository")
                assert result["status"] == "completed"
                times.append((time.perf_counter() - start) * 1000)
        finally:
            store.close()
    ordered = sorted(times)
    return {"kind": "synthetic_orchestration_only", "runs": runs, "model_calls_per_run": 2,
            "median_ms": statistics.median(times), "p95_ms": ordered[min(len(ordered)-1, int(.95 * len(ordered)))],
            "min_ms": min(times), "includes": "workspace lock, SQLite checkpoints/events, one real repository listing",
            "excludes": "inference, network latency, editing, verification; no Loopie performance claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=100)
    args = p.parse_args()
    if not 1 <= args.runs <= 10000:
        p.error("runs must be 1–10000")
    print(json.dumps(asyncio.run(measure(args.runs)), indent=2))
