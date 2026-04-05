"""Normalize Mify upstream HTTP errors into JSON + optional hints."""

from __future__ import annotations

import json
from typing import Any

import httpx
from fastapi import HTTPException


def _attach_model_hint(detail: Any, raw_text: str) -> Any:
    if "Not supported model string" not in raw_text and "Not supported model" not in raw_text:
        return detail
    hint = (
        "Call GET /config/mify to see the exact default_model, default_voice, and default_asr_model "
        "in use. In `.env` set DEFAULT_MODEL, DEFAULT_VOICE (TTS) and DEFAULT_ASR_MODEL (ASR) to IDs "
        "from your Mify admin for each X-Model-Provider-Id, or pass model/voice per request. "
        "TTS errors: fix tts.* + MIFY_PROVIDER_ID; ASR errors: fix asr.* + ASR_PROVIDER_ID."
    )
    if isinstance(detail, dict):
        return {**detail, "hint": hint}
    return {"upstream": detail, "hint": hint}


def http_detail_from_response(response: httpx.Response) -> Any:
    text = response.text or ""
    try:
        detail: Any = response.json()
    except (json.JSONDecodeError, ValueError):
        detail = text
    return _attach_model_hint(detail, text)


def mify_upstream_http_exception(e: httpx.HTTPStatusError) -> HTTPException:
    """Use instead of raw `response.text` so clients get JSON bodies and fix hints."""
    return HTTPException(
        status_code=e.response.status_code,
        detail=http_detail_from_response(e.response),
    )
