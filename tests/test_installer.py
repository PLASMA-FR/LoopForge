"""Installer transactions use real files/links/tar extraction and a mocked pip."""
from __future__ import annotations

import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="loopforge installer ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prefix, self.bin = self.root / "application files", self.root / "command links"
        self.home = self.root / "home"
        self.home.mkdir()
        self.shims = self.root / "shims"
        self.shims.mkdir()
        self.python = self.shims / "test-python"
        # Delegate Python validation/path/extraction/link operations to real Python;
        # simulate only venv/pip to keep tests offline and independent of PyPI.
        self.python.write_text("#!" + sys.executable + "\n" + '''import os, pathlib, sys
if sys.argv[1:3] == ['-m', 'venv']:
    folder = pathlib.Path(sys.argv[3]) / 'bin'
    folder.mkdir(parents=True)
    pip = folder / 'python'
    pip.write_text('#!' + sys.executable + "\\nimport os,sys\\nassert sys.argv[1:4] == ['-m','pip','install']\\nsys.exit(int(os.environ.get('TEST_PIP_FAIL','0')))\\n")
    pip.chmod(0o755)
    cli = folder / 'loopforge'
    cli.write_text('#!/bin/sh\\nprintf "LoopForge test CLI\\\\n"\\n')
    cli.chmod(0o755)
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
        self.python.chmod(0o755)
        curl = self.shims / "curl"
        curl.write_text("#!" + sys.executable + "\n" + '''import os, pathlib, shutil, sys
args = sys.argv[1:]
dest = args[args.index('-o')+1]
assert any(a.startswith('https://codeload.github.com/PLASMA-FR/LoopForge/tar.gz/') for a in args)
shutil.copyfile(os.environ['TEST_ARCHIVE'], dest)
''')
        curl.chmod(0o755)
        self.env = {**os.environ, "HOME": str(self.home), "XDG_CONFIG_HOME": str(self.home / "config"),
                    "PATH": str(self.shims) + os.pathsep + os.environ["PATH"], "TMPDIR": str(self.root)}
        for key in ("LOOPFORGE_REF", "LOOPFORGE_INSTALL_DIR", "LOOPFORGE_BIN_DIR", "LOOPFORGE_PYTHON", "TEST_PIP_FAIL"):
            self.env.pop(key, None)

    def install(self, *extra, download=False):
        args = ["bash", str(ROOT / "install.sh"), "--prefix", str(self.prefix), "--bin-dir", str(self.bin), "--python", str(self.python)]
        if not download:
            args += ["--source", str(ROOT)]
        return subprocess.run([*args, *extra], env=self.env, text=True, capture_output=True, timeout=30)

    def archive(self, members=None):
        archive = self.root / "source.tar.gz"
        if members is None:
            members = {"LoopForge-main/pyproject.toml": (ROOT / "pyproject.toml").read_bytes(),
                       "LoopForge-main/loopforge/cli.py": b"# fixture\n",
                       "LoopForge-main/examples/loopforge.toml": b"profile = 'fast'\n"}
        with tarfile.open(archive, "w:gz") as out:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
        self.env["TEST_ARCHIVE"] = str(archive)
        return archive

    def test_first_install_and_upgrade_preserve_config(self):
        first = self.install()
        self.assertEqual(first.returncode, 0, first.stderr)
        current = self.prefix / "current"
        old = current.resolve()
        self.assertTrue((self.bin / "loopforge").is_symlink())
        cli = subprocess.run([str(self.bin / "loopforge"), "--help"], capture_output=True, text=True)
        self.assertIn("LoopForge test CLI", cli.stdout)
        config = self.home / "config/loopforge/config.toml"
        config.write_text("profile = 'deep'\n")
        second = self.install()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertNotEqual(current.resolve(), old)
        self.assertTrue(old.is_dir())
        self.assertEqual(config.read_text(), "profile = 'deep'\n")
        self.assertFalse(list(self.root.glob("loopforge-download.*")))

    def test_failed_pip_keeps_old_install_and_removes_staging(self):
        self.assertEqual(self.install().returncode, 0)
        old = (self.prefix / "current").resolve()
        self.env["TEST_PIP_FAIL"] = "2"
        failed = self.install()
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual((self.prefix / "current").resolve(), old)
        self.assertEqual(list((self.prefix / "releases").iterdir()), [old])

    def test_unrelated_launcher_is_not_replaced(self):
        self.bin.mkdir()
        launcher = self.bin / "loopforge"
        launcher.write_text("user's command")
        result = self.install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrelated command", result.stderr)
        self.assertEqual(launcher.read_text(), "user's command")

    def test_download_checksum_and_extraction(self):
        archive = self.archive()
        sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        result = self.install("--sha256", sha, "--ref", "main", download=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.bin / "loopforge").exists())

    def test_bad_checksum_does_not_activate(self):
        self.archive()
        result = self.install("--sha256", "0" * 64, download=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256 mismatch", result.stderr)
        self.assertFalse((self.prefix / "current").exists())

    def test_tar_traversal_and_symlink_rejected(self):
        self.archive({"../escaped.txt": b"bad"})
        result = self.install(download=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe archive", result.stderr)
        self.assertFalse((self.root / "escaped.txt").exists())
        archive = self.root / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as out:
            info = tarfile.TarInfo("LoopForge-main/link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc"
            out.addfile(info)
        self.env["TEST_ARCHIVE"] = str(archive)
        result = self.install(download=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsafe archive", result.stderr)

    def test_invalid_flags_fail_before_install(self):
        for args in (("--unknown",), ("--ref",), ("--sha256", "bad")):
            result = self.install(*args)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run(["bash", str(ROOT / "install.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--source", result.stdout)
