from __future__ import annotations

import json


class ContextOverflow(ValueError):
    pass


def token_upper_bound(messages: list[dict]) -> int:
    # UTF-8 bytes conservatively upper-bound byte-BPE tokens, plus message/template
    # overhead. Not a tokenizer count. Deliberately favors small prompts on CPUs.
    return sum(len(json.dumps(m, ensure_ascii=False).encode("utf-8")) + 32 for m in messages) + 256


def assemble(system: str, session: dict, budget: int, protocol: str) -> tuple[list[dict], int]:
    root = [{"role": "system", "content": system}]
    # Keep user constraints verbatim, even after tool history compaction.
    root.extend(session["inputs"][:1])
    turns = session["turns"]
    ledger = {"memory": session.get("memory", ""), "changed_files": session.get("changed", []),
              "shell_used": session.get("shell_used", False),
              "notice": "Older tool turns omitted. Re-read current files before editing; do not infer test success."}
    dropped = 0
    while True:
        messages = root.copy()
        if dropped:
            # Preserve follow-up constraints whose chronological turns were evicted.
            retained = [m for turn in turns[dropped:] for m in turn]
            messages.extend(m for m in session["inputs"][1:] if m not in retained)
            messages.append({"role": "user", "content": "HARNESS STATE: " + json.dumps(ledger, ensure_ascii=False)})
        for turn in turns[dropped:]:
            messages.extend(turn)
        if token_upper_bound(messages) <= budget:
            return messages, dropped
        if dropped >= len(turns):
            raise ContextOverflow("User instructions exceed context budget. Start a new session or increase context_tokens.")
        dropped += 1
