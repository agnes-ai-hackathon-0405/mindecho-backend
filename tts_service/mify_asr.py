"""Call Mify speech-to-text: POST /v1/audio/transcriptions."""

from __future__ import annotations

import base64
import logging
import re

import httpx

from tts_service.settings import settings

logger = logging.getLogger(__name__)

TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"

_DATA_URL_PREFIX = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)


def _normalize_base64(s: str) -> str:
    s = s.strip()
    if _DATA_URL_PREFIX.match(s):
        s = _DATA_URL_PREFIX.sub("", s)
    return s


async def transcribe(
    audio_bytes: bytes,
    *,
    model: str | None = None,
    conversation_id: str | None = None,
    model_request_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """
    Send raw audio bytes (pcm / wav / mp3 / ogg opus) to Mify ASR.
    Returns the upstream JSON object.
    """
    url = settings.mify_base_url.rstrip("/") + TRANSCRIPTIONS_PATH
    headers = {
        "Authorization": f"Bearer {settings.mify_api_key}",
        "X-Model-Provider-Id": settings.asr_provider_id,
        "Content-Type": "application/json",
    }
    if conversation_id is not None and conversation_id != "":
        headers["X-Conversation-Id"] = conversation_id
    if model_request_id:
        headers["X-Model-Request-Id"] = model_request_id
    if user_id:
        headers["X-User-Id"] = user_id

    body = {
        "audio": {"data": base64.b64encode(audio_bytes).decode("ascii")},
        "model": model or settings.default_asr_model,
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        r = await client.post(url, headers=headers, json=body)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            logger.warning("ASR response was not JSON: %s", r.text[:500])
            return {"raw": r.text}


async def transcribe_base64(
    audio_b64: str,
    *,
    model: str | None = None,
    conversation_id: str | None = None,
    model_request_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    normalized = _normalize_base64(audio_b64)
    try:
        raw = base64.b64decode(normalized, validate=False)
    except Exception as e:
        raise ValueError("Invalid base64 audio") from e
    return await transcribe(
        raw,
        model=model,
        conversation_id=conversation_id,
        model_request_id=model_request_id,
        user_id=user_id,
    )
