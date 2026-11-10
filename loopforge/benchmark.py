"""Small reproducible coding evaluation. Real model only; no fabricated speedups."""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .backend import Backend
from .config import Config
from .engine import Engine
from .state import Store
from .tools import command


TASKS = [
    {"name": "fix_binary_search", "file": "solution.py",
     "source": "def find(values, target):\n    lo, hi = 0, len(values)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if values[mid] < target:\n            lo = mid\n        else:\n            hi = mid\n    return lo if lo < len(values) and values[lo] == target else -1\n",
     "task": "Fix find(values, target) in solution.py. Return first matching index or -1, handle empty inputs. Preserve O(log n). Inspect files and verify using python check.py.",
     "check": "from solution import find\nassert find([],1)==-1\nassert find([1],1)==0\nassert find([1,2,2,4],2)==1\nassert find([1,2,3,4],4)==3\nassert find([1,2,3],9)==-1\nprint('PASS')\n"},
    {"name": "implement_intervals", "file": "solution.py",
     "source": "def merge_intervals(items):\n    raise NotImplementedError\n",
     "task": "Implement merge_intervals(items) in solution.py: merge touching/overlapping closed integer intervals, return sorted list of tuples, no mutation of input, handle empty input. Verify with python check.py.",
     "check": "from solution import merge_intervals as f\nassert f([])==[]\nassert f([(3,5),(1,3),(9,10)])==[(1,5),(9,10)]\nx=[(4,8),(1,2),(2,4)]; old=x[:]; assert f(x)==[(1,8)]; assert x==old\nassert f([(1,1),(1,1)])==[(1,1)]\nprint('PASS')\n"},
    {"name": "follow_constraints", "file": "solution.py",
     "source": "def unique(items):\n    return list(set(items))\n",
     "task": "Fix unique(items) in solution.py to preserve first occurrence order and support unhashable values like lists. Do not change check.py. Verify with python check.py.",
     "check": "from solution import unique\nassert unique([3,1,3,2,1])==[3,1,2]\nassert unique([[1],[2],[1]])==[[1],[2]]\nassert unique([])==[]\nprint('PASS')\n"},
]


async def benchmark(config: Config, profiles: list[str], repeats: int) -> dict:
    rows = []
    for repeat in range(repeats):
        for profile in profiles:
            for task in TASKS:
                print(f"Evaluation {profile}/{task['name']} #{repeat+1}", file=sys.stderr)
                with tempfile.TemporaryDirectory(prefix="loopforge-eval-") as tmp:
                    root = Path(tmp)
                    project = root / "project"
                    project.mkdir()
                    (project / task["file"]).write_text(task["source"])
                    (project / "check.py").write_text(task["check"])
                    cfg = replace(config, workspace=str(project), state_dir=str(root / "state"), profile=profile,
                                  allow_shell=True, verification_commands=[f"{sys.executable} check.py"]).validate()
                    backend, store = Backend(cfg), Store(cfg.state_dir)
                    start = time.monotonic()
                    try:
                        result = await Engine(cfg, backend, store).run(task["task"])
                        # Independent validator bytes, stored OUTSIDE model workspace for this run.
                        # Model may edit check.py; evaluator never trusts that copy.
                        checker = "import sys;sys.path.insert(0," + repr(str(project)) + ");" + task["check"]
                        check = await command([sys.executable, "-c", checker], root, 5, 4000)
                        row = {"profile": profile, "task": task["name"], "repeat": repeat,
                               "passed": check["ok"] and result["status"] == "completed",
                               "seconds": time.monotonic() - start, "result": result, "independent_check": check}
                        rows.append(row)
                    finally:
                        await backend.close()
                        store.close()
    summary = []
    for profile in profiles:
        subset = [r for r in rows if r["profile"] == profile]
        success = sum(r["passed"] for r in subset)
        summary.append({"profile": profile, "passed": success, "tasks": len(subset),
                        "success_rate": success / len(subset),
                        "median_seconds": statistics.median(r["seconds"] for r in subset),
                        "seconds_per_success": sum(r["seconds"] for r in subset) / success if success else None,
                        "model_calls": sum(r["result"]["stats"]["model_calls"] for r in subset)})
    return {"model": config.model, "created": time.time(), "summary": summary, "runs": rows,
            "limitations": "Three small Python tasks; not SWE-bench. Includes backend cold/warm effects. Compare on the same machine/backend and inspect diffs. Checks are functional, not a security sandbox."}
