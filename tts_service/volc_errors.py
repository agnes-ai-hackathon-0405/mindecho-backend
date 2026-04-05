"""Map volcengine Python SDK failures to structured HTTP errors."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from fastapi import HTTPException


def _parse_json_from_volc_sdk_message(msg: str) -> dict[str, Any] | None:
    """SDK often raises Exception(str) where str is repr of bytes: b'{\"code\":50400,...}'."""
    msg = msg.strip()
    if "Exception: " in msg:
        msg = msg.split("Exception: ", 1)[-1].strip()

    if msg.startswith("b'") or msg.startswith('b"'):
        try:
            raw = ast.literal_eval(msg)
            if isinstance(raw, (bytes, bytearray)):
                return json.loads(raw.decode("utf-8"))
        except (ValueError, SyntaxError, json.JSONDecodeError, UnicodeDecodeError):
            pass

    start = msg.find("{")
    if start < 0:
        return None
    tail = msg[start:]
    for end in range(len(tail), 0, -1):
        try:
            return json.loads(tail[:end])
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{[^{}]*(?:\"code\"|\{)[^{}]*\}", msg)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def http_exception_from_volc_sdk(exc: BaseException) -> HTTPException:
    """Turn VisualService / Jimeng SDK exceptions into FastAPI HTTPException (never 500)."""
    payload = _parse_json_from_volc_sdk_message(str(exc))
    code = payload.get("code") if payload else None
    msg = (payload.get("message") if payload else None) or str(exc)
    request_id = payload.get("request_id") if payload else None

    code_int: int | None
    try:
        code_int = int(code) if code is not None else None
    except (TypeError, ValueError):
        code_int = None

    # 50400 Access Denied: IAM / product not opened for this key
    if code_int == 50400 or "Access Denied" in str(msg):
        return HTTPException(
            status_code=403,
            detail={
                "error": "volcengine_access_denied",
                "message": msg,
                "code": code,
                "request_id": request_id,
                "hint": (
                    "Open 即梦 AI / 视觉智能 (CV) API access for this Access Key in the Volcengine console "
                    "(IAM permissions or product activation). Keys must match the account that subscribed to Jimeng."
                ),
            },
        )

    return HTTPException(
        status_code=502,
        detail={
            "error": "volcengine_visual_api",
            "message": msg,
            "code": code,
            "request_id": request_id,
            "raw": str(exc)[:2000],
        },
    )
