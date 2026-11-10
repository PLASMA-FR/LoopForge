from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import signal
import stat
import tempfile
from pathlib import Path

from .config import Config
from .protocol import Action, validate_action


READ_ONLY = {"list", "read", "search"}
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".loopforge", "dist", "build"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = (limit - 80) // 2
    return text[:half] + f"\n[truncated {len(text) - 2 * half} characters]\n" + text[-half:]


async def command(command: str | list[str], cwd: Path, timeout: float, limit: int) -> dict:
    argv = ["bash", "--noprofile", "--norc", "-c", command] if isinstance(command, str) else command
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL, start_new_session=True,
        env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb", "NO_COLOR": "1"},
    )
    head = bytearray()
    tail = bytearray()
    total = 0

    async def drain():
        nonlocal total
        assert proc.stdout is not None
        while data := await proc.stdout.read(8192):
            total += len(data)
            if len(head) < limit:
                head.extend(data[:limit - len(head)])
            tail.extend(data)
            if len(tail) > limit:
                del tail[:-limit]

    reader = asyncio.create_task(drain())
    timed_out = False

    def kill():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        async with asyncio.timeout(timeout):
            await proc.wait()
            await asyncio.shield(reader)
    except TimeoutError:
        timed_out = True
        kill()
        await proc.wait()
    finally:
        # Kill surviving descendants on success, cancellation and timeout.
        kill()
        if proc.returncode is None:
            await proc.wait()
        if not reader.done():
            reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
    if total <= limit:
        output = bytes(head).decode("utf-8", "replace")
    else:
        half = limit // 2
        output = (bytes(head[:half]).decode("utf-8", "replace") +
                  f"\n[truncated; {total} output bytes]\n" + bytes(tail[-half:]).decode("utf-8", "replace"))
    return {"ok": proc.returncode == 0 and not timed_out, "exit_code": proc.returncode,
            "timed_out": timed_out, "output": output, "output_bytes": total}


class Tools:
    def __init__(self, config: Config):
        self.config = config
        self.root = Path(config.workspace)
        self.changed: set[str] = set()
        self.shell_used = False
        self.memory = ""
        self.seen_instructions: dict[str, str] = {}

    def instructions(self, path: Path) -> list[dict]:
        found = []
        for parent in reversed(path.parent.parents):
            if parent != self.root and parent.is_relative_to(self.root):
                found.append(parent / "AGENTS.md")
        if path.parent != self.root:
            found.append(path.parent / "AGENTS.md")
        new = []
        for instruction in dict.fromkeys(found):
            if instruction == path or not instruction.is_file():
                continue
            self.path(str(instruction.relative_to(self.root)))
            data = self.bytes(instruction)
            if len(data) > 16000:
                raise ValueError("Nested AGENTS.md exceeds 16 KB instruction limit")
            rel, sha = str(instruction.relative_to(self.root)), digest(data)
            if self.seen_instructions.get(rel) != sha:
                new.append({"path": rel, "content": data.decode("utf-8")})
                self.seen_instructions[rel] = sha
        return new

    def path(self, name: str = ".") -> Path:
        raw = Path(name)
        if raw.is_absolute() or ".." in raw.parts or ".git" in raw.parts:
            raise ValueError("Use workspace-relative paths outside .git")
        path = self.root / raw
        # File tools reject symlinks rather than accidentally rewriting their target.
        current = self.root
        for part in raw.parts:
            current /= part
            if current.is_symlink():
                raise ValueError("Symlinks are not supported by file tools")
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Path escapes workspace")
        if resolved.exists() and not (resolved.is_file() or resolved.is_dir()):
            raise ValueError("Only regular files and directories are supported")
        return resolved

    def bytes(self, path: Path) -> bytes:
        if not path.is_file():
            raise ValueError("Expected regular file")
        with path.open("rb") as f:
            data = f.read(self.config.max_file_bytes + 1)
        if len(data) > self.config.max_file_bytes:
            raise ValueError("File exceeds max_file_bytes; use search or raise the configured limit")
        if b"\0" in data:
            raise ValueError("Binary files are not supported")
        return data

    async def files(self, path: Path, limit: int = 5000) -> list[str]:
        if path.is_file():
            return [str(path.relative_to(self.root))]
        # rg is bounded and obeys ignore files. Never cache stale repository reads.
        if shutil.which("rg"):
            rel = str(path.relative_to(self.root))
            result = await command(["rg", "--files", "--hidden", "-g", "!.git", "-g", "!node_modules",
                                    "-g", "!.venv", "-g", "!.loopforge", "--", rel],
                                   self.root, min(10, self.config.command_timeout), 500_000)
            if result["exit_code"] in (0, 1) and not result["timed_out"] and result["output_bytes"] <= 500_000:
                return sorted(result["output"].splitlines())[:limit]
        # Dependency-free fallback, pruned, with a hard traversal cap.
        def walk():
            found = []
            visited = 0
            for parent, dirs, names in os.walk(path, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(parent) / d).is_symlink())
                for name in sorted(names):
                    visited += 1
                    file = Path(parent) / name
                    if not file.is_symlink() and file.is_file():
                        found.append(str(file.relative_to(self.root)))
                    if len(found) >= limit or visited >= 20000:
                        return found
            return found
        return await asyncio.to_thread(walk)

    def replace(self, path: Path, content: str, sha256: str | None) -> dict:
        encoded = content.encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise ValueError("New file exceeds max_file_bytes")
        mode = 0o644
        if path.exists():
            old = self.bytes(path)
            if sha256 != digest(old):
                raise ValueError("Stale or missing SHA256; read the current file before editing")
            if old == encoded:
                return {"ok": True, "path": str(path.relative_to(self.root)), "changed": False, "sha256": digest(old)}
            mode = stat.S_IMODE(path.stat().st_mode)
            backup = Path(self.config.state_dir) / "blobs" / digest(old)
            backup.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not backup.exists():
                fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as f:
                    f.write(old)
        elif sha256:
            raise ValueError("File no longer exists; re-read workspace")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".loopforge-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(temp, mode)
            # Revalidate immediately before replace (not an OS sandbox or a CAS lock).
            self.path(str(path.relative_to(self.root)))
            if path.exists() and digest(self.bytes(path)) != sha256:
                raise ValueError("File changed during edit; retry after read")
            os.replace(temp, path)
            # Python timestamp-based .pyc validity uses whole-second mtime + size.
            # Fast same-size edits can otherwise keep executing the previous code.
            cache_dir = path.parent / "__pycache__"
            if path.suffix == ".py" and cache_dir.is_dir() and not cache_dir.is_symlink():
                for cache in cache_dir.iterdir():
                    if cache.name.startswith(path.stem + ".") and cache.name.endswith(".pyc"):
                        try:
                            cache.unlink()
                        except OSError:
                            pass
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
        rel = str(path.relative_to(self.root))
        self.changed.add(rel)
        return {"ok": True, "path": rel, "changed": True, "sha256": digest(encoded)}

    async def execute(self, action: Action) -> dict:
        try:
            validate_action(action)
            a = action.arguments
            name = action.name
            if name == "list":
                limit = min(max(a.get("limit", 200), 1), 1000)
                files = await self.files(self.path(a.get("path", ".")), limit + 1)
                return {"ok": True, "files": files[:limit], "truncated": len(files) > limit}
            if name == "read":
                path = self.path(a["path"])
                data = await asyncio.to_thread(self.bytes, path)
                lines = data.decode("utf-8").splitlines()
                start = max(a.get("start", 1), 1)
                end = min(a.get("end", start + 199), start + 999)
                if end < start:
                    raise ValueError("end must be >= start")
                text = "\n".join(f"{i}: {line}" for i, line in enumerate(lines[start - 1:end], start))
                return {"ok": True, "path": a["path"], "sha256": digest(data), "total_lines": len(lines),
                        "content": clip(text, self.config.max_tool_chars), "instructions": self.instructions(path)}
            if name == "search":
                query = a["query"]
                if not query or len(query) > 1000:
                    raise ValueError("query must have 1–1000 characters")
                limit = min(max(a.get("limit", 50), 1), 200)
                path = self.path(a.get("path", "."))
                if shutil.which("rg"):
                    result = await command(["rg", "-n", "-F", "--hidden", "-g", "!.git", "-g", "!node_modules",
                                            "-g", "!.venv", "-g", "!.loopforge", "-m", str(limit),
                                            "--", query, str(path.relative_to(self.root))],
                                           self.root, min(10, self.config.command_timeout), self.config.max_tool_chars)
                    if result["exit_code"] not in (0, 1) or result["timed_out"]:
                        return {**result, "ok": False}
                    return {"ok": True, "matches": result["output"].splitlines()[:limit],
                            "truncated": result["output_bytes"] > self.config.max_tool_chars or len(result["output"].splitlines()) > limit}
                def search(files):
                    matches = []
                    for rel in files:
                        try:
                            lines = self.bytes(self.path(rel)).decode("utf-8").splitlines()
                        except (ValueError, UnicodeError, OSError):
                            continue
                        for i, line in enumerate(lines, 1):
                            if query in line:
                                matches.append(f"{rel}:{i}:{line[:2000]}")
                                if len(matches) >= limit:
                                    return matches
                    return matches
                matches = await asyncio.to_thread(search, await self.files(path))
                return {"ok": True, "matches": matches, "truncated": len(matches) >= limit}
            if name in {"edit", "write"}:
                path = self.path(a["path"])
                instructions = self.instructions(path)
                if instructions:
                    return {"ok": False, "error": "Read these applicable repository instructions, then retry the edit.",
                            "instructions": instructions}
                if name == "edit":
                    old = await asyncio.to_thread(self.bytes, path)
                    text = old.decode("utf-8")
                    if not a["old"] or text.count(a["old"]) != 1:
                        raise ValueError("old must match exactly once; read a more specific range")
                    content = text.replace(a["old"], a["new"], 1)
                else:
                    content = a["content"]
                worker = asyncio.create_task(asyncio.to_thread(self.replace, path, content, a.get("sha256")))
                try:
                    return await asyncio.shield(worker)
                except asyncio.CancelledError:
                    # A thread cannot be forcibly cancelled. Finish the atomic file
                    # operation before the engine releases its workspace lock.
                    await asyncio.gather(worker, return_exceptions=True)
                    raise
            if name == "run":
                if not self.config.allow_shell:
                    raise ValueError("Shell disabled. Use file tools or start with --allow-shell")
                self.shell_used = True
                timeout = max(0.1, min(a.get("timeout", self.config.command_timeout), self.config.command_timeout))
                return await command(a["command"], self.root, timeout, self.config.max_tool_chars)
            if name == "checkpoint":
                self.memory = a["note"][:6000]
                return {"ok": True, "note": self.memory}
            raise ValueError("finish is handled by the engine")
        except (ValueError, OSError, UnicodeError) as e:
            return {"ok": False, "error": str(e)[:1500]}

    async def batch(self, actions: list[Action]) -> list[dict]:
        # Only contiguous read-only groups are concurrent. Never move reads across writes.
        results = []
        index = 0
        semaphore = asyncio.Semaphore(self.config.max_parallel_reads)

        async def read(action):
            async with semaphore:
                return await self.execute(action)

        while index < len(actions):
            end = index
            while end < len(actions) and actions[end].name in READ_ONLY:
                end += 1
            if end > index:
                results.extend(await asyncio.gather(*(read(a) for a in actions[index:end])))
                index = end
            else:
                results.append(await self.execute(actions[index]))
                index += 1
        return results
