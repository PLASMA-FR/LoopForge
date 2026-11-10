from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .config import Config
from .protocol import native_tools, repeated_tail


class BackendError(RuntimeError):
    pass


class RepetitionError(BackendError):
    pass


@dataclass(slots=True)
class Completion:
    message: dict
    usage: dict
    elapsed: float
    finish_reason: str
    first_token_seconds: float | None = None


class Backend:
    def __init__(self, config: Config, client: httpx.AsyncClient | None = None):
        self.config = config
        self.owned = client is None
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.request_timeout, connect=10),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            # Local inference never needs a machine-wide HTTP/SOCKS proxy.
            # For remote backends retain inherited proxy and CA configuration.
            trust_env=urlparse(config.base_url).hostname not in {"127.0.0.1", "localhost", "::1"},
        )

    async def close(self):
        if self.owned:
            await self.client.aclose()

    def headers(self):
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    async def models(self):
        try:
            r = await self.client.get(self.config.base_url.rstrip("/") + "/models", headers=self.headers())
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError) as e:
            raise BackendError(f"Backend model discovery failed: {type(e).__name__}") from e

    async def complete(self, messages: list[dict], max_tokens: int | None = None,
                       tools: bool = True, temperature: float | None = None,
                       top_p: float | None = None) -> Completion:
        payload = {**self.config.extra_body, "model": self.config.model, "messages": messages,
                   "max_tokens": max_tokens or self.config.output_budget,
                   "temperature": self.config.temperature if temperature is None else temperature,
                   "top_p": self.config.top_p if top_p is None else top_p,
                   "stream": True, "stream_options": {"include_usage": True}}
        if not tools:
            payload.pop("response_format", None)
            payload.pop("structured_outputs", None)
        if tools and self.config.protocol == "native":
            payload["tools"] = native_tools()
            payload["tool_choice"] = "auto"
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        started = time.monotonic()
        # Retry only transient transport/status failures BEFORE any generation arrives.
        for attempt in range(3):
            received = False
            content = ""
            reasoning = ""
            calls: dict[int, dict] = {}
            usage = {}
            first_token = None
            finish = None
            wire_bytes = 0
            try:
                async with asyncio.timeout(self.config.request_timeout):
                    async with self.client.stream("POST", url, json=payload, headers=self.headers()) as r:
                        if r.status_code in (429, 502, 503, 504) and attempt < 2:
                            raw = r.headers.get("retry-after", "")
                            try:
                                delay = min(5.0, max(0.0, float(raw)))
                            except ValueError:
                                delay = 0.25 * 2 ** attempt
                            await r.aread()
                            await asyncio.sleep(delay)
                            continue
                        if r.is_error:
                            # Avoid putting backend bodies/credentials into user-visible errors.
                            raise BackendError(f"Backend HTTP {r.status_code}; check URL, model ID and serving logs")
                        if "text/event-stream" not in r.headers.get("content-type", ""):
                            raw = await r.aread()
                            obj = json.loads(raw)
                            choice = obj["choices"][0]
                            if not isinstance(choice["message"], dict):
                                raise BackendError("Backend message must be an object")
                            return Completion(choice["message"], obj.get("usage", {}),
                                              time.monotonic() - started, choice.get("finish_reason") or "stop")
                        async for line in r.aiter_lines():
                            wire_bytes += len(line)
                            if wire_bytes > 4_000_000:
                                raise BackendError("Backend response exceeded 4 MB stream limit")
                            if len(line) > 1_000_000:
                                raise BackendError("Oversized backend event")
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            event = json.loads(data)
                            if "error" in event:
                                raise BackendError("Backend returned an error event")
                            usage = event.get("usage") or usage
                            for choice in event.get("choices", []):
                                if choice.get("index", 0) != 0:
                                    continue
                                delta = choice.get("delta") or {}
                                if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
                                    received = True
                                    if first_token is None:
                                        first_token = time.monotonic() - started
                                content += delta.get("content") or ""
                                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                                for piece in delta.get("tool_calls") or []:
                                    idx = piece.get("index", 0)
                                    if type(idx) is not int or not 0 <= idx < 8:
                                        raise BackendError("Too many backend tool calls")
                                    call = calls.setdefault(idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                                    if piece.get("id"):
                                        call["id"] = piece["id"]
                                    fn = piece.get("function", {})
                                    call["function"]["name"] += fn.get("name") or ""
                                    call["function"]["arguments"] += fn.get("arguments") or ""
                                # Repeated source text inside a JSON edit can be legitimate.
                                prose = not content.lstrip().startswith(("{", "```", "<tool_call>"))
                                if (prose and repeated_tail(content)) or repeated_tail(reasoning):
                                    raise RepetitionError("Stopped exact repeated generation; retry with a different action")
                                if choice.get("finish_reason"):
                                    finish = choice["finish_reason"]
                        if finish is None:
                            raise BackendError("Backend stream ended without a finish reason; discarded partial action")
                        message = {"role": "assistant", "content": content or None}
                        if calls:
                            message["tool_calls"] = [calls[i] for i in sorted(calls)]
                        return Completion(message, usage, time.monotonic() - started, finish, first_token)
            except (httpx.TransportError, TimeoutError) as e:
                if not received and attempt < 2:
                    await asyncio.sleep(0.25 * 2 ** attempt)
                    continue
                raise BackendError(f"Backend transport failed: {type(e).__name__}; partial output discarded") from e
            except (ValueError, KeyError, TypeError, AttributeError, IndexError) as e:
                raise BackendError("Malformed backend response; no action executed") from e
        raise BackendError("Backend unavailable after transient retries")
