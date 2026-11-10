from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from loopforge.backend import Backend, BackendError, Completion, RepetitionError
from loopforge.config import Config, load_config
from loopforge.context import ContextOverflow, assemble, token_upper_bound
from loopforge.engine import Engine
from loopforge.protocol import Action, ProtocolError, parse_actions, repeated_tail, strip_reasoning
from loopforge.server import create_app
from loopforge.state import Store, workspace_lock
from loopforge.tools import Tools, command, digest


def answer(obj, finish="stop"):
    message = obj if isinstance(obj, dict) and "role" in obj else {"role": "assistant", "content": json.dumps(obj) if not isinstance(obj, str) else obj}
    return Completion(message, {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}, 0.01, finish, 0.001)


class ScriptBackend:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    async def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self.script:
            raise AssertionError("Unexpected model call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(messages)
        return item

    async def close(self):
        pass


class HarnessTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.cfg = Config(workspace=str(self.project), state_dir=str(self.root / "state"), allow_shell=True).validate()
        self.store = Store(self.cfg.state_dir)

    async def asyncTearDown(self):
        self.store.close()
        self.tmp.cleanup()

    async def test_exact_edit_stale_and_backup(self):
        path = self.project / "a.py"
        path.write_text("x = 1\n")
        tools = Tools(self.cfg)
        read = await tools.execute(Action("read", {"path": "a.py"}))
        stale = await tools.execute(Action("edit", {"path": "a.py", "old": "1", "new": "2", "sha256": "bad"}))
        self.assertFalse(stale["ok"])
        self.assertEqual(path.read_text(), "x = 1\n")
        valid = await tools.execute(Action("edit", {"path": "a.py", "old": "1", "new": "2", "sha256": read["sha256"]}))
        self.assertTrue(valid["ok"])
        self.assertEqual(path.read_text(), "x = 2\n")
        self.assertTrue((Path(self.cfg.state_dir) / "blobs" / read["sha256"]).exists())

    async def test_workspace_boundaries_and_symlink(self):
        tools = Tools(self.cfg)
        (self.project / "link").symlink_to(self.root)
        for path in ("../outside", "/etc/passwd", ".git/config", "link/file"):
            out = await tools.execute(Action("write", {"path": path, "content": "bad"}))
            self.assertFalse(out["ok"], path)

    async def test_no_partial_write_on_ambiguous_match(self):
        (self.project / "a").write_text("foo foo")
        out = await Tools(self.cfg).execute(Action("edit", {"path": "a", "old": "foo", "new": "bar", "sha256": digest(b"foo foo")}))
        self.assertFalse(out["ok"])
        self.assertEqual((self.project / "a").read_text(), "foo foo")

    async def test_nested_instructions_precede_new_file_edit(self):
        (self.project / "src").mkdir()
        (self.project / "src" / "AGENTS.md").write_text("Use explicit types in this directory.")
        tools = Tools(self.cfg)
        action = Action("write", {"path": "src/a.py", "content": "x: int = 1"})
        first = await tools.execute(action)
        self.assertFalse(first["ok"])
        self.assertIn("explicit types", first["instructions"][0]["content"])
        self.assertFalse((self.project / "src" / "a.py").exists())
        self.assertTrue((await tools.execute(action))["ok"])

    async def test_batch_preserves_mutation_order(self):
        tools = Tools(self.cfg)
        outputs = await tools.batch([Action("write", {"path": "x", "content": "first"}),
                                     Action("read", {"path": "x"}),
                                     Action("edit", {"path": "x", "old": "first", "new": "second", "sha256": digest(b"first")}),
                                     Action("read", {"path": "x"})])
        self.assertTrue(all(o["ok"] for o in outputs))
        self.assertIn("first", outputs[1]["content"])
        self.assertIn("second", outputs[3]["content"])

    async def test_command_bounded_output_timeout_and_shell_opt_in(self):
        out = await command(["python", "-c", "print('x'*1000000)"], self.project, 5, 1000)
        self.assertTrue(out["ok"])
        self.assertLess(len(out["output"]), 1200)
        self.assertGreater(out["output_bytes"], 900000)
        out = await command(["python", "-c", "import time;time.sleep(10)"], self.project, .1, 1000)
        self.assertTrue(out["timed_out"])
        out = await Tools(replace(self.cfg, allow_shell=False)).execute(Action("run", {"command": "echo hi"}))
        self.assertFalse(out["ok"])

    async def test_engine_repairs_then_edits_and_verifies(self):
        (self.project / "calc.py").write_text("def add(a,b): return a-b\n")
        cfg = replace(self.cfg, verification_commands=["python -c 'from calc import add; assert add(2,3)==5'"])
        backend = ScriptBackend([
            answer("not json"), answer({"tool": "read", "args": {"path": "calc.py"}}),
            answer({"tool": "edit", "args": {"path": "calc.py", "old": "a-b", "new": "a+b", "sha256": digest(b"def add(a,b): return a-b\n")}}),
            answer({"tool": "finish", "args": {"summary": "Fixed addition."}}),
        ])
        result = await Engine(cfg, backend, self.store).run("Fix add and test it")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["verified"])
        self.assertEqual(result["stats"]["repairs"], 1)
        self.assertEqual(result["changed_files"], ["calc.py"])
        self.assertIsNone(self.store.get(result["session_id"], cfg.workspace)["pending"])

    async def test_verification_failure_returns_to_model(self):
        cfg = replace(self.cfg, verification_commands=["python -c 'from result import x; assert x==2'"])
        backend = ScriptBackend([
            answer({"tool": "write", "args": {"path": "result.py", "content": "x=1"}}),
            answer({"tool": "finish", "args": {"summary": "done"}}),
            answer({"tool": "edit", "args": {"path": "result.py", "old": "x=1", "new": "x=2", "sha256": digest(b"x=1")}}),
            answer({"tool": "finish", "args": {"summary": "done"}}),
        ])
        result = await Engine(cfg, backend, self.store).run("set x to 2")
        self.assertTrue(result["verified"])
        self.assertIn("verification failed", json.dumps(backend.calls[2][0]))

    async def test_unverified_edits_labelled(self):
        backend = ScriptBackend([answer({"tool": "write", "args": {"path": "x", "content": "hello"}}),
                                 answer({"tool": "finish", "args": {"summary": "done"}})])
        result = await Engine(self.cfg, backend, self.store).run("write hello")
        self.assertFalse(result["verified"])
        self.assertIn("unverified", result["content"])

    async def test_pending_actions_never_replayed(self):
        session = self.store.new(self.cfg.workspace)
        session["pending"] = {"kind": "tools", "actions": [{"tool": "write", "args": {"path": "x", "content": "bad"}}]}
        self.store.save(session)
        backend = ScriptBackend([answer({"tool": "finish", "args": {"summary": "inspected"}})])
        engine = Engine(self.cfg, backend, self.store)
        result = await engine.run("continue", session["id"])
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(len(backend.calls), 0)
        result = await engine.run("inspect and continue", session["id"], recover=True)
        self.assertEqual(result["status"], "completed")
        self.assertFalse((self.project / "x").exists())

    async def test_repeat_failed_action_stops(self):
        cfg = replace(self.cfg, repetition_limit=1)
        backend = ScriptBackend([answer({"tool": "read", "args": {"path": "absent"}})] * 5)
        result = await Engine(cfg, backend, self.store).run("Find file")
        self.assertEqual(result["status"], "stalled")
        self.assertEqual(result["stats"]["tool_calls"], 1)

    async def test_truncated_generation_not_executed(self):
        backend = ScriptBackend([answer({"tool": "write", "args": {"path": "bad", "content": "x"}}, "length"),
                                 answer({"tool": "finish", "args": {"summary": "stopped"}})])
        await Engine(self.cfg, backend, self.store).run("anything")
        self.assertFalse((self.project / "bad").exists())
        self.assertEqual(backend.calls[0][1]["max_tokens"], 2048)
        self.assertEqual(backend.calls[1][1]["max_tokens"], 4096)

    async def test_cancel_kills_child_and_journals_pending(self):
        backend = ScriptBackend([answer({"tool": "run", "args": {"command": "echo $$ > pid; sleep 20"}})])
        task = asyncio.create_task(Engine(self.cfg, backend, self.store).run("wait"))
        for _ in range(100):
            if (self.project / "pid").exists():
                break
            await asyncio.sleep(.01)
        self.assertTrue((self.project / "pid").exists())
        pid = int((self.project / "pid").read_text())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        latest = self.store.recent(self.cfg.workspace)[0]
        saved = self.store.get(latest["id"], self.cfg.workspace)
        self.assertEqual(saved["status"], "cancelled")
        self.assertIsNotNone(saved["pending"])

    async def test_lock_serializes_writers(self):
        order = []
        async def work(i):
            async with workspace_lock(self.cfg.workspace):
                order.append((i, "start"))
                await asyncio.sleep(.05)
                order.append((i, "end"))
        await asyncio.gather(work(1), work(2))
        self.assertEqual(order, [(1, "start"), (1, "end"), (2, "start"), (2, "end")])

    async def test_context_compaction_retains_constraints_and_native_pairs(self):
        session = self.store.new(self.cfg.workspace)
        first = {"role": "user", "content": "Do not remove tests"}
        followup = {"role": "user", "content": "Keep public API unchanged"}
        session["inputs"] = [first, followup]
        session["turns"] = [[followup], [{"role": "assistant", "tool_calls": [{"id": "c1"}], "content": "x" * 6000},
                                       {"role": "tool", "tool_call_id": "c1", "content": "y" * 6000}],
                            [{"role": "user", "content": "recent"}]]
        messages, dropped = assemble("system", session, 2500, "native")
        self.assertGreater(dropped, 0)
        self.assertLessEqual(token_upper_bound(messages), 2500)
        self.assertIn(first, messages)
        self.assertIn(followup, messages)
        self.assertFalse(any(m.get("role") == "tool" for m in messages))

    async def test_api_agent_chat_stream_and_rejections(self):
        backend = ScriptBackend([answer({"tool": "finish", "args": {"summary": "hello"}}),
                                 answer("plain answer"),
                                 answer({"tool": "finish", "args": {"summary": "stream result"}})])
        app = create_app(self.cfg, backend, self.store)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            body = {"model": "loopforge", "messages": [{"role": "user", "content": "hello"}]}
            r = await client.post("/v1/chat/completions", json=body)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()["choices"][0]["message"]["content"], "hello")
            self.assertIn("session_id", r.json()["x_loopforge"])
            r = await client.post("/v1/chat/completions", json={**body, "mode": "chat", "temperature": .3, "top_p": .8})
            self.assertEqual(r.json()["choices"][0]["message"]["content"], "plain answer")
            self.assertEqual(backend.calls[1][1]["top_p"], .8)
            r = await client.post("/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}})
            self.assertIn("data: [DONE]", r.text)
            self.assertIn("stream result", r.text)
            for change in ({"tools": []}, {"n": 2}, {"max_tokens": -2}, {"stream": "yes"}, {"messages": [{"role": "user", "content": []}]}):
                r = await client.post("/v1/chat/completions", json={**body, **change})
                self.assertEqual(r.status_code, 400, r.text)

    async def test_api_auth(self):
        with patch.dict(os.environ, {self.cfg.server_key_env: "secret"}):
            app = create_app(self.cfg, ScriptBackend([]), self.store)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                self.assertEqual((await client.get("/v1/models")).status_code, 401)
                self.assertEqual((await client.get("/healthz")).status_code, 200)
                self.assertEqual((await client.get("/v1/models", headers={"Authorization": "Bearer secret"})).status_code, 200)

    async def test_evaluator_keeps_independent_checks(self):
        from loopforge.benchmark import benchmark
        check = "from solution import x\nassert x==2\n"
        backend = ScriptBackend([
            answer({"tool": "write", "args": {"path": "check.py", "content": "", "sha256": digest(check.encode())}}),
            answer({"tool": "finish", "args": {"summary": "done"}}),
        ])
        fixture = [{"name": "tamper", "file": "solution.py", "source": "x=1\n", "task": "set x to 2", "check": check}]
        with patch("loopforge.benchmark.TASKS", fixture), patch("loopforge.benchmark.Backend", return_value=backend):
            report = await benchmark(self.cfg, ["fast"], 1)
        self.assertTrue(report["runs"][0]["result"]["verified"])
        self.assertFalse(report["runs"][0]["passed"])


class ProtocolTest(unittest.TestCase):
    def test_config_validation_and_profiles(self):
        self.assertEqual(Config(profile="fast").output_budget, 1024)
        self.assertEqual(Config(profile="deep").output_budget, 4096)
        for override in ({"max_steps": 2.5}, {"allow_shell": "false"}, {"request_timeout": float("inf")},
                         {"max_parallel_reads": True}, {"extra_body": {"messages": []}}):
            with self.assertRaises(ValueError):
                Config(**override).validate()
    def test_protocol_formats(self):
        expected = Action("read", {"path": "a"})
        for content in ('{"tool":"read","args":{"path":"a"}}',
                        '```json\n{"tool":"read","args":{"path":"a"}}\n```',
                        '<think>internal</think>{"tool":"read","args":{"path":"a"}}',
                        '<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>'):
            self.assertEqual(parse_actions({"content": content}), [expected])
        native = {"tool_calls": [{"function": {"name": "read", "arguments": '{"path":"a"}'}}]}
        self.assertEqual(parse_actions(native), [expected])
        self.assertEqual(strip_reasoning("private</think>public"), "public")
        self.assertEqual(strip_reasoning("<think>private"), "")

    def test_invalid_json_is_not_repaired_into_code(self):
        for text in ('prose {"tool":"run","args":{"command":"bad"}}', '{"tool":"run",}',
                     '{"calls":[]}', '{"tool":"read","args":null}', '[1]',
                     '{"calls":[{"tool":"finish","args":{}},{"tool":"run","args":{}}]}'):
            with self.assertRaises(ProtocolError, msg=text):
                parse_actions({"content": text})

    def test_repeat_detector(self):
        self.assertTrue(repeated_tail(("abcd" * 24) * 4))
        self.assertFalse(repeated_tail("normal answer"))


class BackendTest(unittest.IsolatedAsyncioTestCase):
    async def test_plain_chat_drops_json_constraints(self):
        seen = []
        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]})
        cfg = Config(extra_body={"response_format": {"type": "json_object"}, "top_k": 20})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            await Backend(cfg, client).complete([], tools=False)
        self.assertNotIn("response_format", seen[0])
        self.assertEqual(seen[0]["top_k"], 20)

    async def test_sse_fragmented_native_calls_and_usage(self):
        seen = []
        events = [
            {"choices": [{"delta": {"reasoning_content": "private"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read", "arguments": '{"path":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58}},
        ]
        def handle(request):
            seen.append(json.loads(request.content))
            text = "".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=text, headers={"Content-Type": "text/event-stream"})
        cfg = Config(protocol="native")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            backend = Backend(cfg, client)
            out = await backend.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(parse_actions(out.message), [Action("read", {"path": "x"})])
        self.assertNotIn("private", json.dumps(out.message))
        self.assertEqual(out.usage["total_tokens"], 58)
        self.assertIn("tools", seen[0])
        self.assertTrue(seen[0]["stream_options"]["include_usage"])

    async def test_truncated_stream_is_discarded(self):
        text = 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\ndata: [DONE]\n\n'
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, text=text, headers={"content-type": "text/event-stream"}))) as client:
            with self.assertRaises(BackendError):
                await Backend(Config(), client).complete([])

    async def test_transient_retry_then_success(self):
        count = 0
        def handle(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(503, headers={"retry-after": "0"})
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            out = await Backend(Config(), client).complete([])
        self.assertEqual(out.message["content"], "ok")
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
