from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from .backend import Backend, BackendError
from .config import load_config
from .engine import Engine
from .state import Store


def parser():
    p = argparse.ArgumentParser(prog="loopforge", description="Linux-first autonomous coding harness for local looped models")
    p.add_argument("--config", help="TOML settings file")
    p.add_argument("--workspace", help="Project directory (default current directory)")
    p.add_argument("--base-url", help="Upstream OpenAI-compatible /v1 URL")
    p.add_argument("--model", help="Exact served upstream model ID")
    p.add_argument("--profile", choices=["fast", "balanced", "deep"])
    p.add_argument("--protocol", choices=["json", "native"])
    p.add_argument("--allow-shell", action="store_true", default=None, help="Authorize bash commands with your account's host permissions")
    p.add_argument("--check", action="append", dest="verification_commands", help="Trusted verification command; repeat for multiple checks")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Complete a single coding task")
    run.add_argument("task", nargs="?", help="Task text, or stdin if omitted")
    run.add_argument("--session", help="Continue a session")
    run.add_argument("--recover", action="store_true", help="Continue interrupted work after inspecting workspace; never replay pending actions")
    run.add_argument("--json", action="store_true", help="Machine-readable result")
    chat = sub.add_parser("chat", help="Interactive coding chat")
    chat.add_argument("--session")
    chat.add_argument("--recover", action="store_true")
    server = sub.add_parser("serve", help="Start the OpenAI-compatible harness endpoint")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", default=8787, type=int)
    sub.add_parser("doctor", help="Inspect backend readiness and local capabilities")
    sub.add_parser("sessions", help="List recent sessions in this workspace")
    inspect = sub.add_parser("inspect", help="Inspect session state and interrupted actions")
    inspect.add_argument("session")
    bench = sub.add_parser("bench", help="Run real-model coding evaluation in temporary workspaces")
    bench.add_argument("--output", default="loopforge-benchmark.json")
    bench.add_argument("--profiles", default="fast,balanced,deep")
    bench.add_argument("--repeats", type=int, default=1)
    from .models import MODEL_REVISION
    install = sub.add_parser("install-model", help="Download Loopie 20B-A2B; optionally install its loop-aware vLLM runtime")
    install.add_argument("name", nargs="?", default="loopie", choices=["loopie", "loopie-20b-a2b"])
    install.add_argument("--directory", help="Model directory; defaults to an XDG data directory per revision")
    install.add_argument("--revision", default=MODEL_REVISION, help="Hugging Face revision (default: pinned checked commit)")
    install.add_argument("--workers", type=int, default=4, help="Concurrent download workers (1–16)")
    install.add_argument("--tools-dir", help="Directory for isolated downloader/runtime environments")
    install.add_argument("--python", default=sys.executable, help="Python executable for separate environments")
    install.add_argument("--with-runtime", action="store_true", help="Also build/install the pinned loop-aware vLLM fork")
    install.add_argument("--dry-run", action="store_true", help="Fetch metadata and print sizes/paths; install/download nothing")
    model_server = sub.add_parser("serve-model", help="Run an installed Loopie vLLM backend on loopback")
    model_server.add_argument("name", nargs="?", default="loopie", choices=["loopie", "loopie-20b-a2b"])
    model_server.add_argument("--directory", help="Model directory (default: pinned Loopie snapshot)")
    model_server.add_argument("--port", type=int, default=8000)
    model_server.add_argument("--context-tokens", type=int, default=16384)
    return p


async def execute(args, config):
    backend, store = Backend(config), Store(config.state_dir)

    async def progress(kind, data):
        if kind == "tools":
            ok = sum(bool(r.get("ok")) for r in data["results"])
            print(f"  {', '.join(data['tools'])}: {ok}/{len(data['results'])} succeeded", file=sys.stderr)
        elif kind in {"repair", "loop_blocked", "verification"}:
            print(f"  {kind}: {json.dumps(data, ensure_ascii=False)[:600]}", file=sys.stderr)

    try:
        if args.command == "sessions":
            print(json.dumps(store.recent(config.workspace), indent=2))
        elif args.command == "inspect":
            print(json.dumps(store.get(args.session, config.workspace), indent=2, ensure_ascii=False))
        elif args.command == "doctor":
            import shutil
            report = {"python": sys.version.split()[0], "workspace": config.workspace, "model": config.model,
                      "protocol": config.protocol, "shell_enabled": config.allow_shell,
                      "rg_available": bool(shutil.which("rg")), "checks": config.verification_commands,
                      "warning": "Backend must implement Loopie's recurrent layers. A stock Qwen3 loader may be incorrect."}
            try:
                report["backend_models"] = await backend.models()
            except BackendError as e:
                report["backend_error"] = str(e)
            print(json.dumps(report, indent=2))
            return 1 if "backend_error" in report else 0
        elif args.command == "bench":
            from .benchmark import benchmark
            profiles = args.profiles.split(",")
            if args.repeats < 1 or args.repeats > 100 or any(p not in {"fast", "balanced", "deep"} for p in profiles):
                raise ValueError("Choose valid profiles and 1–100 repeats")
            report = await benchmark(config, profiles, args.repeats)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report["summary"], indent=2))
            print(f"Report: {Path(args.output).resolve()}")
        elif args.command == "run":
            task = args.task if args.task is not None else sys.stdin.read()
            if not task.strip():
                raise ValueError("Provide a task")
            result = await Engine(config, backend, store).run(task, args.session, args.recover, progress)
            print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else result["content"])
            if not args.json:
                print(f"\nSession: {result['session_id']} | {result['status']} | {result.get('stats', {}).get('elapsed_seconds', 0)}s", file=sys.stderr)
            return 0 if result["status"] == "completed" else 2
        elif args.command == "chat":
            session_id = args.session
            plain = False
            history = []
            recover = args.recover
            print("LoopForge · /new /plain /agent /session /quit · Ctrl-C stops current work")
            while True:
                try:
                    # Prompt only while idle; keep input on main thread so Ctrl-C
                    # cannot leave an executor thread blocked forever on stdin.
                    task = input("you> ")
                except EOFError:
                    break
                if task.strip() in {"/quit", "/exit"}:
                    break
                if task.strip() == "/new":
                    session_id, history = None, []
                    continue
                if task.strip() == "/session":
                    print(session_id or "No session yet")
                    continue
                if task.strip() in {"/plain", "/agent"}:
                    plain = task.strip() == "/plain"
                    print("Conversational chat" if plain else "Autonomous coding")
                    continue
                if not task.strip():
                    continue
                if plain:
                    from .context import token_upper_bound
                    from .protocol import strip_reasoning
                    history.append({"role": "user", "content": task})
                    if token_upper_bound(history) > config.context_tokens - config.output_budget - 512:
                        print("Chat history exceeds context budget; use /new")
                        history.pop()
                        continue
                    answer = await backend.complete(history, tools=False)
                    content = strip_reasoning(answer.message.get("content") or "")
                    history.append({"role": "assistant", "content": content})
                    print("agent> " + content)
                else:
                    result = await Engine(config, backend, store).run(task, session_id, recover, progress)
                    recover = False
                    session_id = result["session_id"]
                    print("agent> " + result["content"])
                    print(f"  [{result['status']}; session {session_id}]")
        return 0
    finally:
        await backend.close()
        store.close()


def main():
    args = parser().parse_args()
    try:
        if args.command in {"install-model", "serve-model"}:
            from .models import install_model, serve_model
            (install_model if args.command == "install-model" else serve_model)(args)
            return
        config = load_config(args.config, **{k: getattr(args, k) for k in (
            "workspace", "base_url", "model", "profile", "protocol", "allow_shell", "verification_commands")})
        if args.command == "serve":
            if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.environ.get(config.server_key_env):
                raise ValueError(f"Set {config.server_key_env} before binding outside loopback")
            import uvicorn
            from .server import create_app
            uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")
            return
        raise SystemExit(asyncio.run(execute(args, config)))
    except (ValueError, OSError, BackendError) as e:
        print(f"loopforge: {e}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        if args.command == "install-model":
            print("\nModel installation stopped. Re-run install-model with the same directory/revision to resume.", file=sys.stderr)
        elif args.command == "serve-model":
            print("\nModel backend stopped.", file=sys.stderr)
        else:
            print("\nStopped. Interrupted actions are recorded; inspect the session before recovery.", file=sys.stderr)
        raise SystemExit(130)
