"""Explicit, user-invoked model installation; never runs inside the agent loop."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

MODEL_ID = "zitian-gao/Loopie-20B-A2B-preview"
MODEL_REVISION = "652d78bb761afc4ac83e454fe71a02beb3f974cf"
HUB_PACKAGE = "huggingface-hub==2.4.0"
RUNTIME_REVISION = "9a9703d9918115cac2c6abfdb84aca411468c16c"
RUNTIME_PACKAGE = f"git+https://github.com/zitian-gao/vllm.git@{RUNTIME_REVISION}"
MANIFEST = ".loopforge-model.json"
SOURCE = ".loopforge-source.json"


def data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))).expanduser() / "loopforge"


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def metadata(revision: str) -> dict:
    if not revision or len(revision) > 200:
        raise ValueError("revision must have 1–200 characters")
    url = f"https://huggingface.co/api/models/{MODEL_ID}/revision/{urllib.parse.quote(revision, safe='')}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Model metadata exceeds 2 MB")
        info = json.loads(raw)
    except (urllib.error.URLError, TimeoutError) as e:
        raise ValueError("Could not read Hugging Face model metadata; check connectivity/revision and retry") from e
    except json.JSONDecodeError as e:
        raise ValueError("Invalid Hugging Face metadata JSON") from e
    if not isinstance(info, dict) or not isinstance(info.get("siblings"), list):
        raise ValueError("Malformed Hugging Face model metadata")
    sha = info.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face did not return a resolved commit SHA")
    files = []
    for item in info.get("siblings", []):
        if not isinstance(item, dict):
            raise ValueError("Malformed model file metadata")
        name, size = item.get("rfilename"), item.get("size")
        if not isinstance(name, str) or type(size) is not int or size < 0:
            raise ValueError("Model metadata is missing file names or sizes")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not re.fullmatch(r"[A-Za-z0-9_./-]+", name) or name.startswith((".loopforge", ".cache/")):
            raise ValueError("Unsafe model file path")
        # This installer supports the safetensors checkpoint, never pickle weights
        # or executable modeling code from the model repository.
        if path.suffix in {".json", ".safetensors", ".txt", ".model", ".jinja", ".tiktoken"} or name == "README.md" or path.name.startswith("LICENSE"):
            files.append({"name": name, "size": size})
    if len({f["name"] for f in files}) != len(files):
        raise ValueError("Duplicate model file metadata")
    names = {f["name"] for f in files}
    if not {"config.json", "tokenizer_config.json", "model.safetensors.index.json"} <= names or not any(n.endswith(".safetensors") for n in names):
        raise ValueError("Repository does not contain the expected Loopie safetensors checkpoint")
    return {"repo_id": MODEL_ID, "revision": sha, "files": files,
            "total_bytes": sum(f["size"] for f in files)}


def _safe_file(directory: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not name:
        raise ValueError("Unsafe model file path")
    path = directory.joinpath(*relative.parts)
    cursor = directory
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"Symlink in model directory: {name}")
    return path


def _json(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink: {path}")
    with path.open("rb") as f:
        raw = f.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError(f"Metadata file exceeds 2 MB: {path.name}")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return obj


def _write_json(path: Path, obj: dict):
    fd, temp = tempfile.mkstemp(prefix=".loopforge-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def _lock(path: Path):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _run(argv: list[str]):
    """Inherit progress output/auth environment; kill descendants on cancellation."""
    process = subprocess.Popen(argv, start_new_session=True)
    try:
        code = process.wait()
        if code:
            raise ValueError(f"Installer subprocess exited with status {code}; see output above. Re-run the command to retry.")
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise


def ensure_environment(root: Path, packages: list[str], python: str, probe: str) -> Path:
    """Build at a permanent path; switch current only after package/probe success."""
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root / ".install.lock"):
        current = root / "current"
        if current.is_symlink():
            active = current.resolve()
            ready = active / ".ready.json"
            settings = _json(ready) if ready.is_file() else {}
            if settings.get("packages") == packages and settings.get("python") == python and (active / "bin/python").is_file():
                return active
        elif current.exists():
            raise ValueError(f"Refusing non-symlink environment pointer: {current}")
        releases = root / "releases"
        releases.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="env-", dir=releases))
        link = root / (".current-" + stage.name)
        try:
            _run([python, "-m", "venv", str(stage)])
            interpreter = str(stage / "bin/python")
            _run([interpreter, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *packages])
            _run([interpreter, "-c", probe])
            _write_json(stage / ".ready.json", {"packages": packages, "python": python})
            link.symlink_to(stage, target_is_directory=True)
            os.replace(link, current)
            return stage
        finally:
            link.unlink(missing_ok=True)
            if not current.is_symlink() or current.resolve() != stage:
                shutil.rmtree(stage, ignore_errors=True)


DOWNLOAD_WORKER = """
import json, sys
from huggingface_hub import snapshot_download
plan = json.loads(sys.argv[1])
snapshot_download(repo_id=plan['repo_id'], revision=plan['revision'],
    local_dir=plan['directory'], allow_patterns=[f['name'] for f in plan['files']],
    max_workers=plan['workers'])
"""


def _destination(directory: Path, plan: dict):
    if directory.is_symlink():
        raise ValueError("Model destination must not be a symlink")
    if not directory.exists():
        return
    source = directory / SOURCE
    if source.exists():
        previous = _json(source)
        if previous.get("repo_id") != MODEL_ID or previous.get("revision") != plan["revision"]:
            raise ValueError("Destination belongs to another model/revision; choose a different --directory")
    else:
        existing = {p.name for p in directory.iterdir()} - {".loopforge-install.lock"}
        if existing:
            raise ValueError("Destination is not an empty or previously managed Loopie download; choose another --directory")
    for item in plan["files"]:
        _safe_file(directory, item["name"])


def validate_download(directory: Path, plan: dict):
    for item in plan["files"]:
        path = _safe_file(directory, item["name"])
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Missing or incomplete model file: {item['name']}. Re-run install-model to resume.")
    config = _json(directory / "config.json")
    if config.get("model_type") != "qwen3_moe" or config.get("layer_loop_times") != 2 or config.get("model_loop_times") != 1:
        raise ValueError("Downloaded config does not match the supported Loopie layer-loop architecture")
    weight_map = _json(directory / "model.safetensors.index.json").get("weight_map")
    names = {item["name"] for item in plan["files"] if item["name"].endswith(".safetensors")}
    if not isinstance(weight_map, dict) or not weight_map or any(not isinstance(v, str) or v not in names for v in weight_map.values()):
        raise ValueError("Safetensors index refers to unsupported or missing weight shards")


def install_model(args) -> dict:
    if sys.platform != "linux":
        raise ValueError("Model installation currently supports Linux")
    if not 1 <= args.workers <= 16:
        raise ValueError("workers must be 1–16")
    if args.with_runtime and not args.dry_run and shutil.which("git") is None:
        raise ValueError("Install git before using --with-runtime (the pinned vLLM fork is built from Git source)")
    info = metadata(args.revision)
    directory = _path(args.directory or data_home() / "models/Loopie-20B-A2B-preview" / info["revision"])
    tools_dir = _path(args.tools_dir or data_home() / "model-tools")
    _destination(directory, info)
    missing = sum(item["size"] for item in info["files"]
                  if not (directory / item["name"]).is_file() or (directory / item["name"]).stat().st_size != item["size"])
    parent = directory
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    plan = {**info, "directory": str(directory), "tools_dir": str(tools_dir), "workers": args.workers,
            "remaining_bytes_estimate": missing, "free_bytes": free,
            "with_runtime": args.with_runtime, "runtime_revision": RUNTIME_REVISION if args.with_runtime else None,
            "note": "Original BF16 weights, about 42.2 GB. This is not a quantized/32-GB laptop runtime. vLLM source builds need additional disk, build tools and suitable hardware."}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    if free < missing + 1024 ** 3:
        raise ValueError(f"Insufficient model disk space: need approximately {(missing + 1024**3) / 1024**3:.1f} GiB free; available {free / 1024**3:.1f} GiB")
    print(f"Installing {MODEL_ID} at {info['revision']}\nDestination: {directory}\nCheckpoint: {info['total_bytes'] / 1e9:.1f} GB; remaining estimate: {missing / 1e9:.1f} GB", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    with _lock(directory / ".loopforge-install.lock"):
        _destination(directory, info)
        _write_json(directory / SOURCE, {"repo_id": MODEL_ID, "revision": info["revision"]})
        helper = ensure_environment(tools_dir / "huggingface", [HUB_PACKAGE, "httpx[socks]>=0.27,<1"], args.python,
                                    "from huggingface_hub import snapshot_download; assert callable(snapshot_download)")
        _run([str(helper / "bin/python"), "-c", DOWNLOAD_WORKER, json.dumps(plan)])
        validate_download(directory, plan)
        record = {"repo_id": MODEL_ID, "revision": info["revision"], "directory": str(directory),
                  "files": info["files"], "total_bytes": info["total_bytes"], "installed_at": time.time(), "runtime": None}
        previous = directory / MANIFEST
        if previous.is_file():
            record["runtime"] = _json(previous).get("runtime")
        # The model is complete even if optional runtime installation later fails.
        _write_json(previous, record)
        if args.with_runtime:
            print("Installing the pinned loop-aware vLLM source fork in a separate environment...", flush=True)
            runtime = ensure_environment(tools_dir / "vllm", [RUNTIME_PACKAGE], args.python,
                "from pathlib import Path; import importlib.util; s=importlib.util.find_spec('vllm'); "
                "assert s and s.submodule_search_locations; p=Path(next(iter(s.submodule_search_locations)))/'model_executor/models/qwen3_moe.py'; "
                "assert 'layer_loop_times' in p.read_text(), 'vLLM lacks layer-loop handling'")
            record["runtime"] = {"environment": str(runtime), "revision": RUNTIME_REVISION}
            _write_json(previous, record)
    print(f"Model installed: {directory}")
    if record["runtime"]:
        print("Start the local backend:\n  " + shlex.join(["loopforge", "serve-model", "loopie", "--directory", str(directory)]))
    else:
        print("Weights downloaded. Add the optional runtime with:\n  " + shlex.join(["loopforge", "install-model", "loopie", "--directory", str(directory), "--revision", info["revision"], "--tools-dir", str(tools_dir), "--with-runtime"]))
    return record


def serve_model(args):
    directory = _path(args.directory or data_home() / "models/Loopie-20B-A2B-preview" / MODEL_REVISION)
    if not 1 <= args.port <= 65535 or args.context_tokens < 1024:
        raise ValueError("port must be 1–65535 and context-tokens must be at least 1024")
    manifest = directory / MANIFEST
    if not manifest.is_file():
        raise ValueError("Model is not installed here. Run loopforge install-model loopie first.")
    record = _json(manifest)
    if record.get("repo_id") != MODEL_ID or not record.get("runtime"):
        raise ValueError("A compatible runtime is not installed. Re-run install-model loopie --with-runtime.")
    validate_download(directory, record)
    executable = Path(record["runtime"]["environment"]) / "bin/vllm"
    if not executable.is_file():
        raise ValueError("Installed runtime is missing; re-run install-model loopie --with-runtime")
    argv = [str(executable), "serve", str(directory), "--served-model-name", MODEL_ID,
            "--host", "127.0.0.1", "--port", str(args.port), "--max-model-len", str(args.context_tokens)]
    print("Starting backend: " + shlex.join(argv), flush=True)
    _run(argv)
