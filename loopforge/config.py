from __future__ import annotations

import os
import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlparse


@dataclass(slots=True)
class Config:
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "zitian-gao/Loopie-20B-A2B-preview"
    api_key_env: str = "LOOPFORGE_UPSTREAM_KEY"
    server_key_env: str = "LOOPFORGE_SERVER_KEY"
    workspace: str = "."
    state_dir: str = "~/.local/state/loopforge"
    profile: str = "balanced"
    protocol: str = "json"
    context_tokens: int = 16384
    max_output_tokens: int = 4096
    max_steps: int = 40
    max_total_tokens: int = 80000
    max_seconds: float = 1200
    request_timeout: float = 180
    command_timeout: float = 60
    max_tool_chars: int = 12000
    max_file_bytes: int = 1_000_000
    max_parallel_reads: int = 4
    repetition_limit: int = 3
    temperature: float = 0.6
    top_p: float = 0.95
    allow_shell: bool = False
    verification_commands: list[str] = field(default_factory=list)
    # Backend-specific parameters are opt-in: no invented latent-loop controls.
    extra_body: dict = field(default_factory=dict)

    def validate(self) -> Config:
        for name in ("base_url", "model", "api_key_env", "server_key_env", "workspace", "state_dir", "profile", "protocol"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.allow_shell) is not bool:
            raise ValueError("allow_shell must be boolean")
        self.workspace = str(Path(self.workspace).expanduser().resolve())
        self.state_dir = str(Path(self.state_dir).expanduser().resolve())
        if not Path(self.workspace).is_dir():
            raise ValueError("workspace must be an existing directory")
        if self.profile not in {"fast", "balanced", "deep"}:
            raise ValueError("profile must be fast, balanced, or deep")
        if self.protocol not in {"json", "native"}:
            raise ValueError("protocol must be json or native")
        if urlparse(self.base_url).scheme not in {"http", "https"}:
            raise ValueError("base_url must use http(s)")
        for name in ("context_tokens", "max_output_tokens", "max_steps", "max_total_tokens",
                     "max_tool_chars",
                     "max_file_bytes", "max_parallel_reads", "repetition_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_seconds", "request_timeout", "command_timeout", "temperature", "top_p"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if name not in {"temperature", "top_p"} and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_tool_chars < 512 or self.max_file_bytes < 512:
            raise ValueError("file and tool limits must be at least 512")
        if self.context_tokens < self.max_output_tokens + 4096:
            raise ValueError("context_tokens must reserve at least 4096 input tokens")
        if not 0 <= self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("invalid sampling settings")
        if not isinstance(self.extra_body, dict):
            raise ValueError("extra_body must be a table")
        reserved = {"model", "messages", "tools", "tool_choice", "stream", "max_tokens",
                    "max_completion_tokens", "temperature", "top_p"}
        if reserved & self.extra_body.keys():
            raise ValueError("extra_body cannot override protocol or budget fields")
        if not isinstance(self.verification_commands, list) or any(
            not isinstance(c, str) or not c.strip() for c in self.verification_commands
        ):
            raise ValueError("verification_commands must be a list of nonempty strings")
        if self.verification_commands and not self.allow_shell:
            raise ValueError("verification_commands requires allow_shell=true")
        return self

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")

    @property
    def output_budget(self) -> int:
        return min(self.max_output_tokens, {"fast": 1024, "balanced": 2048, "deep": 4096}[self.profile])


def load_config(path: str | None = None, **overrides) -> Config:
    data = {}
    if path:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    known = {f.name for f in fields(Config)}
    if data.keys() - known:
        raise ValueError(f"unknown settings: {sorted(data.keys() - known)}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**data).validate()
