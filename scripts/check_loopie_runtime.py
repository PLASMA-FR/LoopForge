#!/usr/bin/env python3
"""Static preflight for a LOCAL vLLM source tree. Does not load weights/GPU code."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("vllm_source", help="Path to reviewed vLLM source checkout")
p.add_argument("--model-config", help="Optional local Loopie config.json")
args = p.parse_args()
source = Path(args.vllm_source) / "vllm/model_executor/models/qwen3_moe.py"
text = source.read_text(encoding="utf-8")
report = {"source": str(source.resolve()), "layer_loop_handling_present": "layer_loop_times" in text,
          "limitation": "Static indicator only; does not prove kernel correctness or validate loaded server."}
if args.model_config:
    cfg = json.loads(Path(args.model_config).read_text())
    report["model"] = {k: cfg.get(k) for k in ("model_type", "layer_loop_times", "model_loop_times", "use_qk_norm")}
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["layer_loop_handling_present"] else 1)
