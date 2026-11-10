"""Real TCP -> uvicorn -> harness -> HTTP backend -> filesystem -> check -> response."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NetworkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.project = root / "project"
        self.project.mkdir()
        (self.project / "value.py").write_text("x=1\n")
        self.state = root / "state"
        self.mode = "fix"
        self.count = 0
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.count += 1
                if owner.mode == "sleep":
                    obj = {"tool": "run", "args": {"command": "echo $$ > child.pid; sleep 30"}}
                elif "tools" not in payload and not any("LoopForge" in (m.get("content") or "") for m in payload["messages"]):
                    obj = None
                elif owner.count == 1:
                    obj = {"tool": "read", "args": {"path": "value.py"}}
                elif owner.count == 2:
                    obj = {"tool": "edit", "args": {"path": "value.py", "old": "x=1", "new": "x=2", "sha256": hashlib.sha256(b"x=1\n").hexdigest()}}
                else:
                    obj = {"tool": "finish", "args": {"summary": "Fixed x."}}
                text = json.dumps(obj) if obj is not None else "Hello from chat."
                event = {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]}
                usage = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
                wire = ("data: " + json.dumps(event) + "\n\ndata: " + json.dumps(usage) + "\n\ndata: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(wire)))
                self.end_headers()
                self.wfile.write(wire)

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.thread.start()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        cfg = root / "config.toml"
        values = {"workspace": str(self.project), "state_dir": str(self.state), "allow_shell": True,
                  "base_url": f"http://127.0.0.1:{self.upstream.server_port}/v1",
                  "verification_commands": [f'{sys.executable} -c "from value import x; assert x == 2"']}
        cfg.write_text("\n".join(k + " = " + json.dumps(v) for k, v in values.items()))
        env = {k: v for k, v in os.environ.items() if k not in {"LOOPFORGE_SERVER_KEY", "LOOPFORGE_UPSTREAM_KEY"}}
        env["NO_PROXY"] = "127.0.0.1,localhost"
        self.log = (root / "server.log").open("w+")
        self.proc = subprocess.Popen([sys.executable, "-m", "loopforge", "--config", str(cfg), "serve", "--port", str(self.port)],
                                     cwd=ROOT, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.addCleanup(self.cleanup)
        self.base = f"http://127.0.0.1:{self.port}"
        for _ in range(100):
            try:
                with urllib.request.urlopen(self.base + "/healthz", timeout=.2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                time.sleep(.05)
        self.log.seek(0)
        self.fail("Server failed to start: " + self.log.read())

    def cleanup(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.thread.join(2)
        self.log.close()
        self.tmp.cleanup()

    def request(self, body):
        request = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(request, timeout=10)

    def test_real_network_coding_and_conversational_stream(self):
        with self.request({"model": "loopforge", "messages": [{"role": "user", "content": "Change x to 2"}]}) as response:
            result = json.load(response)
        self.assertEqual(result["choices"][0]["message"]["content"], "Fixed x.")
        self.assertTrue(result["x_loopforge"]["verified"])
        self.assertEqual((self.project / "value.py").read_text(), "x=2\n")
        self.assertEqual(result["usage"]["total_tokens"], 360)
        with self.request({"model": "loopforge", "mode": "chat", "stream": True,
                           "messages": [{"role": "user", "content": "Hi"}]}) as response:
            text = response.read().decode()
        self.assertIn("Hello from chat.", text)
        self.assertIn("data: [DONE]", text)

    def test_real_disconnect_cancels_shell(self):
        self.mode = "sleep"
        response = self.request({"model": "loopforge", "stream": True,
                                 "messages": [{"role": "user", "content": "Start a shell job"}]})
        self.assertIn(b"working", response.readline())
        for _ in range(200):
            if (self.project / "child.pid").exists():
                break
            time.sleep(.01)
        self.assertTrue((self.project / "child.pid").exists())
        pid = int((self.project / "child.pid").read_text())
        response.close()
        for _ in range(200):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.01)
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
