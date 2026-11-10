from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from loopforge import models
from loopforge.cli import parser


def fixture():
    files = {
        "config.json": json.dumps({"model_type": "qwen3_moe", "layer_loop_times": 2, "model_loop_times": 1}).encode(),
        "tokenizer_config.json": b"{}",
        "model.safetensors.index.json": b'{"weight_map":{"layer.weight":"model-00001-of-00001.safetensors"}}',
        "model-00001-of-00001.safetensors": b"fake test weights",
    }
    info = {"repo_id": models.MODEL_ID, "revision": models.MODEL_REVISION,
            "files": [{"name": n, "size": len(v)} for n, v in files.items()],
            "total_bytes": sum(map(len, files.values()))}
    return info, files


class ModelInstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / "model"
        self.tools = self.root / "tools"
        self.info, self.files = fixture()
        self.stdout = io.StringIO()
        self.args = SimpleNamespace(name="loopie", directory=str(self.directory), tools_dir=str(self.tools),
                                    revision=models.MODEL_REVISION, workers=2, with_runtime=False,
                                    dry_run=False, python=sys.executable)

    def materialize(self):
        self.directory.mkdir(exist_ok=True)
        for name, content in self.files.items():
            (self.directory / name).write_bytes(content)

    def install(self, worker=None, environments=None):
        calls = []
        def run(argv):
            calls.append(argv)
            if worker:
                worker(argv)
            elif argv[2] == models.DOWNLOAD_WORKER:
                self.materialize()
        with patch.object(models, "metadata", return_value=self.info), \
             patch.object(models, "ensure_environment", side_effect=environments or [self.tools / "hf"]), \
             patch.object(models, "_run", side_effect=run), contextlib.redirect_stdout(self.stdout):
            result = models.install_model(self.args)
        return result, calls

    def test_download_writes_complete_manifest_and_pins_snapshot(self):
        result, calls = self.install()
        self.assertIsNone(result["runtime"])
        self.assertEqual(result["revision"], models.MODEL_REVISION)
        self.assertTrue((self.directory / models.MANIFEST).is_file())
        payload = json.loads(calls[0][-1])
        self.assertEqual(payload["revision"], models.MODEL_REVISION)
        self.assertEqual(payload["workers"], 2)
        self.assertIn("--with-runtime", self.stdout.getvalue())

    def test_dry_run_only_fetches_metadata(self):
        self.args.dry_run = True
        with patch.object(models, "metadata", return_value=self.info), \
             patch.object(models, "ensure_environment") as env, patch.object(models, "_run") as run, \
             contextlib.redirect_stdout(self.stdout):
            result = models.install_model(self.args)
        self.assertFalse(self.directory.exists())
        env.assert_not_called()
        run.assert_not_called()
        self.assertEqual(json.loads(self.stdout.getvalue())["revision"], models.MODEL_REVISION)
        self.assertEqual(result["remaining_bytes_estimate"], self.info["total_bytes"])

    def test_download_failure_leaves_resumable_source_marker(self):
        def fail(argv):
            self.directory.joinpath("config.json").write_bytes(self.files["config.json"])
            raise ValueError("network interrupted")
        with self.assertRaisesRegex(ValueError, "interrupted"):
            self.install(worker=fail)
        self.assertTrue((self.directory / models.SOURCE).is_file())
        self.assertFalse((self.directory / models.MANIFEST).exists())
        result, _ = self.install()
        self.assertEqual(result["repo_id"], models.MODEL_ID)

    def test_incomplete_shard_is_not_marked_complete(self):
        def incomplete(argv):
            self.materialize()
            (self.directory / "model-00001-of-00001.safetensors").write_bytes(b"truncated")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.install(worker=incomplete)
        self.assertFalse((self.directory / models.MANIFEST).exists())

    def test_disk_check_prevents_download(self):
        with patch.object(models, "metadata", return_value=self.info), \
             patch.object(models.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
             patch.object(models, "ensure_environment") as env:
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                models.install_model(self.args)
        env.assert_not_called()
        self.assertFalse(self.directory.exists())

    def test_unrelated_directory_and_changed_revision_are_refused(self):
        self.directory.mkdir()
        (self.directory / "user.txt").write_text("leave me")
        with self.assertRaisesRegex(ValueError, "not an empty"):
            self.install()
        (self.directory / "user.txt").unlink()
        (self.directory / models.SOURCE).write_text(json.dumps({"repo_id": models.MODEL_ID, "revision": "a" * 40}))
        with self.assertRaisesRegex(ValueError, "another model/revision"):
            self.install()

    def test_with_runtime_and_local_serve_command(self):
        self.args.with_runtime = True
        runtime = self.tools / "vllm"
        (runtime / "bin").mkdir(parents=True)
        (runtime / "bin/vllm").touch()
        result, _ = self.install(environments=[self.tools / "hf", runtime])
        self.assertEqual(result["runtime"]["revision"], models.RUNTIME_REVISION)
        args = SimpleNamespace(directory=str(self.directory), port=8123, context_tokens=8192)
        with patch.object(models, "_run") as run, contextlib.redirect_stdout(self.stdout):
            models.serve_model(args)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:3], [str(runtime / "bin/vllm"), "serve", str(self.directory)])
        self.assertIn("127.0.0.1", argv)
        self.assertEqual(argv[argv.index("--served-model-name")+1], models.MODEL_ID)
        self.assertEqual(argv[-1], "8192")

    def test_runtime_failure_preserves_completed_model(self):
        self.args.with_runtime = True
        with self.assertRaisesRegex(ValueError, "build failed"):
            self.install(environments=[self.tools / "hf", ValueError("build failed")])
        self.assertIsNone(json.loads((self.directory / models.MANIFEST).read_text())["runtime"])

    def test_environment_failure_keeps_previous_pointer(self):
        root = self.tools / "runtime"
        root.mkdir(parents=True)
        old = root / "old"
        old.mkdir()
        (root / "current").symlink_to(old, target_is_directory=True)
        with patch.object(models, "_run", side_effect=ValueError("pip failed")):
            with self.assertRaisesRegex(ValueError, "pip failed"):
                models.ensure_environment(root, ["fixture==1"], sys.executable, "pass")
        self.assertEqual((root / "current").resolve(), old)
        self.assertEqual(list((root / "releases").iterdir()), [])

    def test_environment_success_is_reused_without_pip(self):
        root = self.tools / "helper"
        def run(argv):
            if argv[1:3] == ["-m", "venv"]:
                interpreter = Path(argv[3]) / "bin/python"
                interpreter.parent.mkdir(parents=True)
                interpreter.touch()
        with patch.object(models, "_run", side_effect=run) as called:
            first = models.ensure_environment(root, ["fixture==1"], sys.executable, "pass")
            self.assertEqual(called.call_count, 3)
            second = models.ensure_environment(root, ["fixture==1"], sys.executable, "pass")
            self.assertEqual(called.call_count, 3)
        self.assertEqual(first, second)
        self.assertEqual((root / "current").resolve(), first)

    def test_missing_model_runtime_and_bad_worker_count(self):
        self.args.workers = 0
        with self.assertRaisesRegex(ValueError, "workers"):
            models.install_model(self.args)
        with self.assertRaisesRegex(ValueError, "not installed"):
            models.serve_model(SimpleNamespace(directory=str(self.directory), port=8000, context_tokens=4096))

    def test_file_symlink_rejected(self):
        self.directory.mkdir()
        (self.directory / models.SOURCE).write_text(json.dumps({"repo_id": models.MODEL_ID, "revision": models.MODEL_REVISION}))
        external = self.root / "outside"
        external.write_text("private")
        (self.directory / "config.json").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.install()
        self.assertEqual(external.read_text(), "private")


class MetadataTest(unittest.TestCase):
    def request(self, body):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return json.dumps(body).encode()
        return patch.object(models.urllib.request, "urlopen", return_value=Response())

    def test_metadata_filters_code_and_pickle(self):
        info, _ = fixture()
        siblings = [{"rfilename": item["name"], "size": item["size"]} for item in info["files"]]
        siblings += [{"rfilename": "model.bin", "size": 500}, {"rfilename": "modeling.py", "size": 100}]
        with self.request({"sha": models.MODEL_REVISION, "siblings": siblings}):
            result = models.metadata("main")
        self.assertEqual(result["files"], info["files"])

    def test_untrusted_metadata_paths_rejected(self):
        for name in ("../config.json", "/etc/config.json", "config[0].json", ".loopforge-source.json"):
            with self.request({"sha": models.MODEL_REVISION, "siblings": [{"rfilename": name, "size": 1}]}):
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    models.metadata("main")

    def test_cli_flags_and_model_commands_bypass_workspace_setup(self):
        args = parser().parse_args(["install-model", "loopie", "--dry-run", "--with-runtime"])
        self.assertTrue(args.dry_run)
        self.assertTrue(args.with_runtime)
        self.assertEqual(args.revision, models.MODEL_REVISION)
        from loopforge.cli import main
        with patch.object(sys, "argv", ["loopforge", "install-model", "loopie", "--dry-run"]), \
             patch("loopforge.cli.load_config") as cfg, patch.object(models, "install_model") as install:
            main()
        cfg.assert_not_called()
        install.assert_called_once()


if __name__ == "__main__":
    unittest.main()
