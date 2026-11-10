# LoopForge

A small, Linux-first autonomous coding harness for **Loopie-20B-A2B-preview** and
other text models served through an OpenAI-compatible backend. Includes terminal
coding chat, conversational chat, one-shot tasks, and a local Chat Completions API.
Python 3.11+; three runtime dependencies; no second LLM or embedding service.

This is runnable source, not a model fine-tune. It reduces orchestration waste and
recovers from common agent errors. It does **not** promise a particular tokens/s,
eliminate model mistakes, or claim to be the fastest harness. A real-model
evaluation command is included so settings can be selected on your hardware.

## Quick start on Linux

Install for your current user with Python 3.11+ and curl:

```bash
curl -fsSL https://raw.githubusercontent.com/PLASMA-FR/LoopForge/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
loopforge --workspace ~/code/my-project doctor
loopforge --workspace ~/code/my-project --allow-shell chat

# Or start the OpenAI-compatible endpoint on 127.0.0.1:8787:
loopforge --workspace ~/code/my-project --allow-shell serve
```

The installer creates a dedicated venv in `~/.local/share/loopforge`, links the CLI
into `~/.local/bin`, and preserves config at `~/.config/loopforge/config.toml`.
Existing configuration is used when passed with `--config`; it is not loaded
automatically. Model weights and the inference backend are installed separately.
The installer runs pip to install Python dependencies; it does not need sudo.
On Debian/Ubuntu, install `python3-venv` if Python cannot create virtual environments.

Running the same installer updates the harness. Each install builds at its permanent
path before switching the active symlink, preserving older versions and leaving
the active installation untouched if pip or its smoke check fails. An existing
unrelated `loopforge` command is never overwritten. Old versions remain under
`~/.local/share/loopforge/releases` until you remove them. Shell profiles are not edited.

Inspect the script before running it, or select a reviewed commit:

```bash
curl -fsSL https://raw.githubusercontent.com/PLASMA-FR/LoopForge/main/install.sh -o install.sh
bash install.sh --help
# Pin BOTH the downloaded installer URL and --ref to the reviewed commit SHA.
# --sha256 optionally verifies the downloaded source archive, not the installer.
bash install.sh --ref COMMIT_SHA --prefix "$HOME/.local/share/loopforge" \
  --bin-dir "$HOME/.local/bin"
```

`--python /path/to/python3.12` selects an interpreter. `--source /path/to/checkout`
installs local source without downloading the archive. To pass options through a
pipeline, use `curl -fsSL URL | bash -s -- --ref COMMIT_SHA`. `LOOPFORGE_REF`,
`LOOPFORGE_INSTALL_DIR`, `LOOPFORGE_BIN_DIR`, and `LOOPFORGE_PYTHON` also set defaults;
command-line options take precedence. Downloads require HTTPS and extraction
rejects escaping paths, symlinks and special files. This does not authenticate a
moving Git branch; use reviewed commit refs and independently obtained checksums
when reproducibility is required.

For development, clone the repository or start in the extracted source directory,
which contains `pyproject.toml`:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
source .venv/bin/activate
cp examples/loopforge.toml loopforge.toml

# Start your compatible Loopie inference server separately on port 8000.
loopforge --config loopforge.toml --workspace ~/code/my-project doctor

# Interactive coding chat; file tools work without shell access.
loopforge --config loopforge.toml --workspace ~/code/my-project chat

# Autonomous coding with commands and a project-specific verification command.
loopforge --config loopforge.toml --workspace ~/code/my-project \
  --allow-shell --check 'python -m unittest discover -s tests -v' \
  run 'Find and fix the failing tests. Preserve the public API.'
```

Use your project's actual test/build command. `--allow-shell` is a one-time
authorization for unattended commands under your Linux account. There are no
per-command confirmation prompts. File tools confine paths to the workspace;
**bash is not sandboxed**. Use a disposable worktree/container or a dedicated
Unix account when isolation is needed. Existing uncommitted changes are not reset.

Global flags go **before** `chat`, `run`, `serve`, etc. Set `--base-url` to the
upstream `/v1` URL and `--model` to its exact served model ID. Set upstream auth
through `LOOPFORGE_UPSTREAM_KEY`; credentials are not written into configuration.
`requirements.lock` records the dependency versions used for the included tests;
use `pip install -r requirements.lock` before installing the package when exact
reproduction is needed. Remote SOCKS proxies additionally need `httpx[socks]`.
Loopback inference connections bypass machine-wide proxy environment variables.

## Loopie backend requirements

The checked checkpoint is
[`zitian-gao/Loopie-20B-A2B-preview`](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview),
revision `652d78bb761afc4ac83e454fe71a02beb3f974cf` (checked 2026-11-10).
Its card emphasizes mathematical/scientific reasoning. The config uses
`layer_loop_times=2`, `model_loop_times=1`, and QK normalization on a `qwen3_moe`
backbone. Its generation settings are temperature **0.6**, top-p **0.95**,
top-k **20**. This harness preserves those sampling defaults where supported.

The model card links custom [vLLM](https://github.com/zitian-gao/vllm) and
[Transformers](https://github.com/zitian-gao/transformers) forks. Use a reviewed,
pinned runtime build that actually implements its repeated layers and cache
layout. A stock runtime recognizing the name `qwen3_moe` is insufficient evidence
that it executes this checkpoint correctly. The checked vLLM source contains
explicit `layer_loop_times` handling. This is a static observation, not a GPU
correctness test. `scripts/check_loopie_runtime.py /path/to/vllm-checkout` checks
for that indicator without importing GPU libraries.

After installing a compatible backend in its **own** environment, the model card's
basic serving command is:

```bash
vllm serve zitian-gao/Loopie-20B-A2B-preview
```

Follow that runtime's current installation/hardware instructions. No inference
fork or weights are installed by this harness. Do not reduce latent layer loops
to gain speed: that changes the model computation and may break quality/caching.
The JSON protocol requires ordinary text generation, so a backend-native tool
parser is optional. Native mode requires the backend to emit proper tool calls.
If your backend supports constrained JSON decoding, you may add
`response_format = {type = "json_object"}` inside `[extra_body]` to reduce malformed
responses. This is opt-in because compatibility with Loopie's custom runtime
must be checked. Plain chat strips `response_format` and `structured_outputs`.

**For a 32 GB laptop:** start with `examples/laptop.toml` and an already working,
architecture-compatible endpoint. Roughly 21B BF16 weights alone exceed 32 GB;
2B active parameters do not mean only 2B parameters must reside in memory. A
compatible quantized/offloaded runtime or a remote endpoint may be necessary.
Loopie GGUF, llama.cpp, Intel GPU offload and speculative decoding support were
not validated here. The harness cannot supply that missing backend support.

## Chat modes

```bash
loopforge --config loopforge.toml --workspace ~/code/my-project --allow-shell chat
```

The default terminal chat can inspect/edit the project and use authorized tools.
`/plain` switches to conversational chat without tools. `/agent` switches back.
`/new` starts fresh history; `/session` prints the persistent coding-session ID;
`/quit` exits. Plain-chat history lives in memory. Coding sessions are persistent.
Ctrl-C cancels current work and exits; shell process groups are killed.

Continue a coding session:

```bash
loopforge --workspace ~/code/my-project sessions
loopforge --workspace ~/code/my-project inspect SESSION_ID
loopforge --workspace ~/code/my-project --allow-shell run --session SESSION_ID 'Continue the fix'
```

If a process stopped during a tool batch, changes may have partially executed.
Inspect the project and session first. `run --session SESSION_ID --recover 'Inspect
the current state and continue'` clears the uncertainty marker and starts from
observations; it never replays pending actions. Recovery is also available on
`chat --session SESSION_ID --recover`. A finished session can receive follow-ups.

## OpenAI-compatible endpoint

```bash
export LOOPFORGE_SERVER_KEY='choose-a-long-random-local-key'
loopforge --config loopforge.toml --workspace ~/code/my-project \
  --allow-shell --check 'python -m unittest discover -s tests -v' \
  serve --host 127.0.0.1 --port 8787
```

Supported endpoints: `GET /healthz`, `GET /v1/models`, and
`POST /v1/chat/completions`. Model alias: `loopforge`.

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $LOOPFORGE_SERVER_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"loopforge","messages":[{"role":"user","content":"Inspect this repository and identify its entry point."}]}'
```

The standard answer is in `choices[0].message.content`. `x_loopforge` adds session,
verification, status, changed-file and timing details. `usage` aggregates internal
model calls, so it may be much larger than the final answer. Missing backend usage
is estimated and labeled `usage_estimated=true`.

| Request setting | Behavior |
| --- | --- |
| `mode: "agent"` (default) | Executes the coding harness against the server's configured workspace |
| `mode: "chat"` | Stateless conversational completion; send history in `messages`; no tools |
| `stream: true` | Valid SSE chunks, heartbeat comments, finish event and `[DONE]` |
| `stream_options: {"include_usage": true}` | Adds the final usage chunk |
| `session_id: "..."` | Continue a coding session; send **only new** messages |
| `recover: true` | Acknowledge inspected interrupted work; never replay actions |
| `max_tokens` or `max_completion_tokens` | Per-call cap, 128 up to the server configuration's maximum |
| `temperature`, `top_p` | Override sampling for this request within validated ranges |

Streaming currently buffers the answer until the agent finishes, in both modes;
heartbeats keep the connection active. Internal reasoning/tool payloads are not
exposed as assistant text. Disconnects cancel work and kill shell process groups.
This endpoint implements a **text-only Chat Completions subset**: `n=1`, no caller
tools, images, `/v1/responses`, embeddings, logprobs, or arbitrary OpenAI parameters.
Unsupported parameters return errors rather than being silently ignored. Coding
clients that require the Responses API cannot connect directly. Ordinary Chat
Completions clients can use:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="your-local-key", timeout=1300)
response = client.chat.completions.create(
    model="loopforge",
    messages=[{"role": "user", "content": "Fix the parser and verify it."}],
    extra_body={"mode": "agent"},
)
print(response.choices[0].message.content)
```

Both credentials and workspace permissions belong to the server operator.
Workspace writers serialize across sessions/processes under the same Unix account.
Plain chat permits two concurrent requests. Non-loopback CLI binds require an API
key; use your own TLS/reverse proxy for remote access. A caller cannot select an
arbitrary workspace, enable shell, or supply verification commands via the API.

## Optimizations and recovery

| Failure/cost | Implementation |
| --- | --- |
| Long prompts slow local prefill | Bounded file reads, literal `rg` search, byte-based context upper bounds, whole-turn compaction |
| Repeated schema/thinking overhead | Small stable tool vocabulary, persistent HTTP connections, stable prompt prefix until compaction, no LLM summarizer |
| Many read round-trips | Up to eight calls in a response; contiguous read-only groups execute concurrently |
| Stale or hallucinated edits | Read SHA256 required for existing files; exact unique replacement; atomic writes; old bytes saved |
| Malformed output | Strict JSON/schema validation and bounded repair feedback; no speculative execution of partial JSON |
| Backend tool parser missing | Default JSON action protocol; optional `--protocol native` |
| Repeated actions/reasoning | Signature counters, stall termination, exact repetitive-stream abort; full request budget reserved for unmetered aborts |
| Failed tests | Operator-selected checks run before completion; failures become repair evidence |
| Crash/cancellation | SQLite WAL checkpoints and pending-action journal; no automatic replay of uncertain effects |
| Runaway work | Per-call, total-token, step, wall-time, command-time and output limits |
| Python same-size fast edits | Relevant `.pyc` entries invalidated after built-in file edits |
| Nested repository rules | Applicable nested `AGENTS.md` surfaced on read or before a built-in edit |

These address general coding-agent failure modes. There is no verified evidence
here that Loopie uniquely suffers each one, and the harness cannot guarantee
instruction-following or correctness. File protections are operational safeguards,
not a security boundary against hostile processes or filesystem races. Bash can
bypass file-tool restrictions. Timed-out commands can leave partial edits.

Context accounting uses conservative UTF-8 byte bounds plus template allowance,
not the model tokenizer. User instructions are retained verbatim; old tool turns
are dropped together with their native tool replies. Checkpoint memory and
changed-file paths survive compaction; file contents must be re-read. Large user
instructions produce an explicit context error. Configure a context no larger
than the actual upstream limit. The advertised 131072 position capacity is not a
measured long-context coding guarantee.
Without `rg`, the fallback walker prunes common generated directories but does
not implement `.gitignore` semantics. Install ripgrep for large repositories.

`verified=true` means configured commands returned zero immediately before
completion. It is evidence about those checks, not a proof of correctness or a
tamper-proof grading system. The model can edit project tests when shell/writes
are enabled. With no checks configured, edited work is explicitly marked
unverified. No commands are guessed for arbitrary projects.

## Select speed settings using evidence

| Profile | Initial output budget | Intended starting point |
| --- | ---: | --- |
| fast | 1024 tokens | Small, straightforward edits; cheap short action cycles |
| balanced | 2048 tokens | Default coding tasks |
| deep | 4096 tokens | Harder reasoning; allow longer backend timeouts |

All are limited by `max_output_tokens`. A truncated action doubles its next-call
budget up to that hard cap, without executing the partial action. These are tuning presets, not
measured quality optima. If the model spends its entire budget in reasoning,
increase the cap or use deep; repeated truncation terminates with a clear error.
Truncation never forces execution of an incomplete action. Fast may take longer
overall if it causes extra repair turns. Keep temperature 0.6 initially; do not
assume greedy sampling is better for this checkpoint.

```bash
loopforge --config loopforge.toml bench --profiles fast,balanced,deep \
  --repeats 3 --output loopforge-benchmark.json
```

This really calls your backend and runs three Python tasks in temporary projects.
It reports success rate, median wall time, total time per success, model calls,
token usage and model/tool timing. Independent final functional checks are stored
outside the model workspace in evaluator memory. Runs are sequential; cold/warm
and ordering effects matter. It is a small local evaluation, not SWE-bench or
proof of safe behavior. The benchmark authorizes shell access in its temporary
projects; it is not OS isolation.

Backend-side prefix caching can reuse stable prefixes when supported. Enable it
only with a loop-aware KV implementation and compare results. Quantization,
speculative decoding and GPU offload must be validated for this architecture
before use. No GPU acceleration, cache speedup, model quality improvement, or
real Loopie throughput was measured in the build environment.

## Tests and synthetic overhead

```bash
python -m unittest discover -s tests -v
python scripts/measure_overhead.py --runs 100
```

The tests use deterministic/mocked models plus actual filesystem operations,
shell execution, checks and API handling. The overhead script measures the
harness with a scripted zero-latency backend; it measures orchestration only.
Its result must not be presented as Loopie inference performance.

## Optional systemd user service

`examples/loopforge.service` expects an installation at
`~/.local/share/loopforge/venv` and config at `~/.config/loopforge/config.toml`:

```bash
python3 -m venv ~/.local/share/loopforge/venv
~/.local/share/loopforge/venv/bin/pip install .
mkdir -p ~/.config/loopforge ~/.config/systemd/user
cp examples/loopforge.toml ~/.config/loopforge/config.toml
# Edit config.toml: use an absolute workspace path and set your checks/access.
cp examples/loopforge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now loopforge.service
journalctl --user -u loopforge.service -f
```

An optional `~/.config/loopforge/environment` file can hold environment-variable
assignments for upstream/server keys; keep it mode 600. The service does not
install or manage the model backend and remains loopback-only by default.

Sessions/logs and pre-edit blobs default to `~/.local/state/loopforge`. They may
contain source code, tool output, task text and snippets of local secrets read by
the model; protect that directory. `inspect` shows recorded state. Blobs are
content-addressed original bytes, not an automatic rollback system. Use Git for
review/reverts; no commits, pushes, or resets happen automatically. Cross-host
session synchronization, Windows support and concurrent mutations from other
accounts/editors are not implemented.

MIT licensed. Model/runtime licenses and hardware requirements are separate.
