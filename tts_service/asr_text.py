"""Extract plain text from Mify / Volcengine ASR JSON (shape varies by provider)."""

from __future__ import annotations

import json
from typing import Any


def extract_transcript_text(asr: Any) -> str | None:
    if asr is None:
        return None
    if isinstance(asr, str):
        t = asr.strip()
        return t or None
    if not isinstance(asr, dict):
        return None

    payload = asr.get("payload")
    if isinstance(payload, str) and payload.strip():
        try:
            inner = json.loads(payload)
            t = extract_transcript_text(inner)
            if t:
                return t
        except json.JSONDecodeError:
            pass

    for key in ("text", "transcript", "result_text", "asr_text"):
        v = asr.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    for path in (
        ("data", "text"),
        ("data", "transcript"),
        ("data", "result", "text"),
        ("result", "text"),
        ("Result", "text"),
    ):
        cur: Any = asr
        for k in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        if isinstance(cur, str) and cur.strip():
            return cur.strip()

    for container_key in ("data", "result", "Result"):
        block = asr.get(container_key)
        if isinstance(block, dict):
            utt = block.get("utterances") or block.get("Utterances")
            if isinstance(utt, list) and utt:
                parts: list[str] = []
                for u in utt:
                    if isinstance(u, dict):
                        tx = u.get("text") or u.get("Text")
                        if isinstance(tx, str) and tx.strip():
                            parts.append(tx.strip())
                if parts:
                    return "".join(parts)

    return None
