from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import replace

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .backend import Backend, BackendError
from .config import Config
from .context import token_upper_bound
from .engine import Engine
from .protocol import strip_reasoning
from .state import Store


def error(message: str, status: int = 400, kind: str = "invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": kind, "code": None}}, status_code=status)


def create_app(config: Config, backend: Backend | None = None, store: Store | None = None) -> FastAPI:
    own_backend, own_store = backend is None, store is None
    backend = backend or Backend(config)
    store = store or Store(config.state_dir)
    plain_slots = asyncio.Semaphore(2)

    @asynccontextmanager
    async def lifespan(app):
        yield
        if own_backend:
            await backend.close()
        if own_store:
            store.close()

    app = FastAPI(title="LoopForge", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        key = os.environ.get(config.server_key_env, "")
        if request.url.path != "/healthz" and key:
            auth = request.headers.get("authorization", "")
            if not hmac.compare_digest(auth.encode(), ("Bearer " + key).encode()):
                return error("Invalid API key", 401, "authentication_error")
        return await call_next(request)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "loopforge", "object": "model", "created": 0, "owned_by": "local"}]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 1_000_000:
                    return error("Request exceeds 1 MB", 413)
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object")
            allowed = {"model", "messages", "stream", "stream_options", "max_tokens", "max_completion_tokens",
                       "temperature", "top_p", "n", "user", "mode", "session_id", "recover"}
            unknown = body.keys() - allowed
            if unknown:
                raise ValueError(f"Unsupported parameters: {sorted(unknown)}. Client tools, images, Responses API and logprobs are not implemented.")
            if body.get("n", 1) != 1:
                raise ValueError("Only n=1 is supported")
            if body.get("model", "loopforge") not in {"loopforge", config.model}:
                return error("Unknown model; use loopforge", 404)
            mode = body.get("mode", "agent")
            if mode not in {"agent", "chat"}:
                raise ValueError("mode must be agent or chat")
            for key in ("stream", "recover"):
                if key in body and type(body[key]) is not bool:
                    raise ValueError(f"{key} must be boolean")
            if "session_id" in body and not isinstance(body["session_id"], str):
                raise ValueError("session_id must be a string")
            stream_options = body.get("stream_options", {})
            if not isinstance(stream_options, dict) or stream_options.keys() - {"include_usage"}:
                raise ValueError("stream_options supports only include_usage")
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty array")
            for m in messages:
                if not isinstance(m, dict) or m.keys() - {"role", "content", "name"} or m.get("role") not in {"system", "user", "assistant"} or not isinstance(m.get("content"), str):
                    raise ValueError("Only text system/user/assistant messages are supported")
            messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            opts = {}
            if "max_tokens" in body and "max_completion_tokens" in body:
                raise ValueError("Use only one max token parameter")
            cap = body.get("max_tokens", body.get("max_completion_tokens", config.max_output_tokens))
            if type(cap) is not int or not 128 <= cap <= config.max_output_tokens:
                raise ValueError(f"max_tokens must be 128–{config.max_output_tokens}")
            opts["max_output_tokens"] = cap
            for key in ("temperature", "top_p"):
                if key in body:
                    if type(body[key]) not in (int, float):
                        raise ValueError(f"{key} must be a number")
                    opts[key] = body[key]
            cfg = replace(config, **opts).validate()
            if token_upper_bound(messages) > cfg.context_tokens - cfg.output_budget - 4096:
                raise ValueError("Input too large for configured context budget")
            if mode == "chat" and (body.get("session_id") or body.get("recover")):
                raise ValueError("Chat mode is stateless; send conversation history in messages")
        except (ValueError, TypeError) as e:
            return error(str(e))

        async def execute():
            if mode == "agent":
                return await Engine(cfg, backend, store).run("", body.get("session_id"), body.get("recover", False), inputs=messages)
            async with plain_slots:
                answer = await backend.complete(messages, max_tokens=cfg.output_budget, tools=False,
                                                temperature=cfg.temperature, top_p=cfg.top_p)
                return {"content": strip_reasoning(answer.message.get("content") or ""), "usage": answer.usage,
                        "status": "completed" if answer.finish_reason != "length" else "budget_exhausted",
                        "finish_reason": answer.finish_reason}

        ident, created = "chatcmpl-" + uuid.uuid4().hex, int(time.time())

        def envelope(result, streaming=False):
            finish = "stop" if result["status"] == "completed" else "length"
            obj = {"id": ident, "object": "chat.completion.chunk" if streaming else "chat.completion",
                   "created": created, "model": "loopforge", "choices": [],
                   "x_loopforge": {k: v for k, v in result.items() if k not in {"content", "usage", "finish_reason"}}}
            if streaming:
                obj["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": result["content"]}, "finish_reason": None}]
            else:
                obj["choices"] = [{"index": 0, "message": {"role": "assistant", "content": result["content"]},
                                   "finish_reason": finish, "logprobs": None}]
                obj["usage"] = result.get("usage", {})
            return obj

        if body.get("stream", False):
            async def generate():
                task = asyncio.create_task(execute())
                try:
                    yield ": loopforge working\n\n"
                    while not task.done():
                        done, _ = await asyncio.wait({task}, timeout=5)
                        if not done:
                            yield ": loopforge working\n\n"
                    result = task.result()
                    yield "data: " + json.dumps(envelope(result, True), ensure_ascii=False) + "\n\n"
                    ending = {"id": ident, "object": "chat.completion.chunk", "created": created, "model": "loopforge",
                              "choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if result["status"] == "completed" else "length"}]}
                    yield "data: " + json.dumps(ending) + "\n\n"
                    if stream_options.get("include_usage"):
                        yield "data: " + json.dumps({**ending, "choices": [], "usage": result.get("usage", {})}) + "\n\n"
                    yield "data: [DONE]\n\n"
                except (BackendError, ValueError) as e:
                    yield "data: " + json.dumps({"error": {"message": str(e), "type": "server_error"}}) + "\n\n"
                    yield "data: [DONE]\n\n"
                finally:
                    if not task.done():
                        task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            return StreamingResponse(generate(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        task = asyncio.create_task(execute())
        try:
            while not task.done():
                await asyncio.wait({task}, timeout=0.2)
                if await request.is_disconnected():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                    return error("Client disconnected; run cancelled", 499)
            return JSONResponse(envelope(task.result()))
        except (BackendError, ValueError) as e:
            return error(str(e), 502, "server_error")
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    return app
