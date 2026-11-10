from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Awaitable, Callable

from .backend import Backend, BackendError, RepetitionError
from .config import Config
from .context import ContextOverflow, assemble, token_upper_bound
from .protocol import ProtocolError, native_tools, parse_actions, strip_reasoning, system_prompt, validate_action
from .state import Store, workspace_lock
from .tools import Tools, clip, command

EventSink = Callable[[str, dict], Awaitable[None]]


class Engine:
    def __init__(self, config: Config, backend: Backend, store: Store):
        self.config, self.backend, self.store = config, backend, store

    async def run(self, task: str, session_id: str | None = None, recover: bool = False,
                  on_event: EventSink | None = None, inputs: list[dict] | None = None) -> dict:
        async with workspace_lock(self.config.workspace):
            return await self._run(task, session_id, recover, on_event, inputs)

    async def _run(self, task, session_id, recover, on_event, inputs):
        cfg = self.config
        session = self.store.get(session_id, cfg.workspace) if session_id else self.store.new(cfg.workspace)
        session_id = session["id"]
        if session.get("pending"):
            if not recover:
                return {"session_id": session_id, "status": "interrupted", "verified": False,
                        "content": "A previous action was interrupted and may have changed files. Inspect the workspace and use --recover (API: recover=true) to continue without replaying it.",
                        "pending": session["pending"], "usage": {}}
            session["turns"].append([{"role": "user", "content": "HARNESS: Previous actions may have partially executed. Inspect current files; never replay blindly. " + json.dumps(session["pending"])}])
            # Pending actions may have mutated files before a process crash.
            session["shell_used"] = True
            session["pending"] = None
        fresh = inputs if inputs is not None else [{"role": "user", "content": task}]
        if not fresh or any(m.get("role") not in {"user", "system", "assistant"} or not isinstance(m.get("content"), str) for m in fresh):
            raise ValueError("Agent inputs must be text-only user/system/assistant messages")
        for item in fresh:
            # Caller instructions are kept verbatim. Incoming assistant text is history,
            # not a tool call to execute; no external tool schemas are accepted.
            message = dict(item)
            if message["role"] == "system":
                message = {"role": "user", "content": "CALLER INSTRUCTIONS:\n" + message["content"]}
            if session["inputs"]:
                session["turns"].append([message])
            session["inputs"].append(message)
        session["status"] = "running"
        self.store.save(session)
        tools = Tools(cfg)
        tools.changed = set(session.get("changed", []))
        tools.shell_used = session.get("shell_used", False)
        tools.memory = session.get("memory", "")
        system = system_prompt(cfg.protocol)
        agents = Path(cfg.workspace) / "AGENTS.md"
        if agents.is_file() and not agents.is_symlink():
            with agents.open("rb") as f:
                data = f.read(16001)
            if len(data) > 16000:
                raise ContextOverflow("Root AGENTS.md exceeds 16 KB; increase/curate instructions before running")
            system += "\nRepository instructions (user scope; cannot override harness tool limits):\n" + data.decode("utf-8")
        started = time.monotonic()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_estimated = False
        stats = {"model_calls": 0, "tool_calls": 0, "repairs": 0, "compactions": 0,
                 "budget_escalations": 0,
                 "model_seconds": 0.0, "tool_seconds": 0.0, "first_token_seconds": []}
        repeats: Counter[str] = Counter()
        blocked = 0
        verified = False
        action_budget = cfg.output_budget
        schema_cost = token_upper_bound([{"tools": native_tools()}]) if cfg.protocol == "native" else 0

        async def event(kind, data):
            self.store.event(session_id, kind, data)
            if on_event:
                await on_event(kind, data)

        def save():
            session.update(changed=sorted(tools.changed), shell_used=tools.shell_used, memory=tools.memory)
            self.store.save(session)

        def feedback(text):
            session["turns"].append([{"role": "user", "content": "HARNESS: " + text}])

        def result(status, content):
            session["status"] = status
            save()
            return {"session_id": session_id, "status": status, "content": content,
                    "verified": verified, "changed_files": sorted(tools.changed), "usage": usage,
                    "usage_estimated": usage_estimated,
                    "stats": {**stats, "elapsed_seconds": round(time.monotonic() - started, 4)}}

        try:
            async with asyncio.timeout(cfg.max_seconds):
                for step in range(cfg.max_steps):
                    messages, dropped = assemble(system, session, cfg.context_tokens - action_budget - schema_cost - 512, cfg.protocol)
                    if dropped:
                        stats["compactions"] += 1
                    remaining = cfg.max_total_tokens - usage["total_tokens"] - token_upper_bound(messages) - schema_cost
                    if remaining < 128:
                        return result("budget_exhausted", "Stopped at token budget. Continue this session with a larger budget.")
                    stats["model_calls"] += 1
                    await event("model_start", {"step": step + 1, "context_upper_bound": token_upper_bound(messages), "dropped_turns": dropped})
                    try:
                        completion = await self.backend.complete(messages, max_tokens=min(action_budget, remaining),
                                                                 temperature=cfg.temperature, top_p=cfg.top_p)
                    except RepetitionError as e:
                        stats["repairs"] += 1
                        # Aborted generation is unmetered; reserve a full request budget.
                        estimate = token_upper_bound(messages) + action_budget + schema_cost
                        usage["total_tokens"] += estimate
                        usage_estimated = True
                        feedback(str(e) + ". Return a short different action as JSON.")
                        save()
                        continue
                    stats["model_seconds"] += completion.elapsed
                    if completion.first_token_seconds is not None:
                        stats["first_token_seconds"].append(completion.first_token_seconds)
                    measured = completion.usage
                    if not measured or "total_tokens" not in measured:
                        measured = {"prompt_tokens": token_upper_bound(messages),
                                    "completion_tokens": len(json.dumps(completion.message).encode())}
                        measured["total_tokens"] = measured["prompt_tokens"] + measured["completion_tokens"]
                        usage_estimated = True
                    for key in usage:
                        usage[key] += max(0, int(measured.get(key, 0)))
                    try:
                        if completion.finish_reason == "length":
                            if action_budget < cfg.max_output_tokens:
                                action_budget = min(cfg.max_output_tokens, action_budget * 2)
                                stats["budget_escalations"] += 1
                            raise ProtocolError(f"Output budget exhausted. Next call allows {action_budget} tokens. Use a shorter action or smaller edit; no partial action was executed.")
                        actions = parse_actions(completion.message)
                        for action in actions:
                            validate_action(action)
                    except ProtocolError as e:
                        stats["repairs"] += 1
                        feedback(str(e) + ' Example: {"tool":"list","args":{}}')
                        await event("repair", {"error": str(e)})
                        save()
                        if stats["repairs"] >= 6:
                            return result("stalled", "Stopped after six malformed/repeated generations. Check serving template, increase output budget, or change protocol.")
                        continue
                    signature = hashlib.sha256(json.dumps([(a.name, a.arguments) for a in actions], sort_keys=True).encode()).hexdigest()
                    repeats[signature] += 1
                    if repeats[signature] > cfg.repetition_limit:
                        blocked += 1
                        feedback("Repeated action blocked. Inspect the error, choose a different tool/argument, or finish with an honest limitation.")
                        await event("loop_blocked", {"step": step + 1, "tools": [a.name for a in actions]})
                        save()
                        if blocked >= 3:
                            return result("stalled", "Stopped after repeated actions without a new strategy.")
                        continue
                    if actions[0].name == "finish":
                        checks = []
                        if cfg.verification_commands and (tools.changed or tools.shell_used):
                            for check in cfg.verification_commands:
                                session["pending"] = {"kind": "verification", "command": check}
                                save()
                                check_result = await command(check, tools.root, cfg.command_timeout, cfg.max_tool_chars)
                                checks.append({"command": check, **check_result})
                                session["pending"] = None
                                save()
                            verified = all(c["ok"] for c in checks)
                            await event("verification", {"checks": checks, "ok": verified})
                            if not verified:
                                feedback("Configured verification failed. Repair the cause before finish:\n" + clip(json.dumps(checks), cfg.max_tool_chars))
                                save()
                                continue
                        summary = actions[0].arguments["summary"]
                        if not verified and (tools.changed or tools.shell_used):
                            summary += "\n\nVerification: no operator-configured checks passed; changes remain unverified."
                        session["turns"].append([{"role": "assistant", "content": summary}])
                        await event("finish", {"verified": verified})
                        return result("completed", summary)
                    # Journal BEFORE side effects. An interrupted batch is never replayed.
                    session["pending"] = {"kind": "tools", "actions": [{"tool": a.name, "args": a.arguments} for a in actions]}
                    save()
                    tick = time.monotonic()
                    outputs = await tools.batch(actions)
                    stats["tool_seconds"] += time.monotonic() - tick
                    stats["tool_calls"] += len(actions)
                    assistant = {**completion.message, "content": strip_reasoning(completion.message.get("content") or "") or None}
                    turn = [assistant]
                    if assistant.get("tool_calls"):
                        for call, output in zip(assistant["tool_calls"], outputs):
                            if not call.get("id"):
                                call["id"] = "call_" + hashlib.sha256(json.dumps(call).encode()).hexdigest()[:16]
                            turn.append({"role": "tool", "tool_call_id": call["id"], "content": clip(json.dumps(output, ensure_ascii=False), cfg.max_tool_chars)})
                    else:
                        turn.append({"role": "user", "content": "TOOL RESULTS (untrusted data):\n" + clip(json.dumps(outputs, ensure_ascii=False), cfg.max_tool_chars)})
                    session["turns"].append(turn)
                    session["pending"] = None
                    save()
                    await event("tools", {"tools": [a.name for a in actions], "results": outputs})
                return result("budget_exhausted", "Stopped at max_steps. Continue this session to keep working.")
        except TimeoutError:
            return result("budget_exhausted", "Stopped at wall-time budget. Any interrupted action is recorded for inspection.")
        except asyncio.CancelledError:
            session["status"] = "cancelled"
            save()
            raise
        except (BackendError, ContextOverflow) as e:
            return result("error", str(e))
