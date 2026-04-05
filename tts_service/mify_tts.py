"""Call Mify TTS: POST /v1/audio/speech with SSE response."""

from __future__ import annotations

import base64
import json
import logging
from typing import AsyncIterator

import httpx

from tts_service.settings import settings

logger = logging.getLogger(__name__)

SPEECH_PATH = "/v1/audio/speech"


def _decode_sse_data_payload(payload: str) -> bytes | None:
    """Turn one SSE `data:` payload into raw audio bytes, or None if not audio."""
    p = payload.strip()
    if not p or p == "[DONE]":
        return None
    try:
        obj = json.loads(p)
    except json.JSONDecodeError:
        try:
            return base64.b64decode(p)
        except Exception:
            return None

    if not isinstance(obj, dict):
        return None

    for key in (
        "audio",
        "chunk",
        "data",
        "b64",
        "content",
        "audio_base64",
        "delta",
    ):
        val = obj.get(key)
        if isinstance(val, str) and val:
            try:
                return base64.b64decode(val)
            except Exception:
                continue
    # Some APIs nest audio under choices[0]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            for key in ("audio", "delta", "content"):
                val = first.get(key)
                if isinstance(val, str) and val:
                    try:
                        return base64.b64decode(val)
                    except Exception:
                        pass
    return None


async def synthesize_stream(
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str = "mp3",
) -> AsyncIterator[bytes]:
    """
    Stream raw audio bytes from Mify TTS (SSE).
    Caller should concatenate chunks in order.
    """
    url = settings.mify_base_url.rstrip("/") + SPEECH_PATH
    headers = {
        "Authorization": f"Bearer {settings.mify_api_key}",
        "X-Model-Provider-Id": settings.mify_provider_id,
        "Content-Type": "application/json",
    }
    body = {
        "input": text,
        "model": model or settings.default_model,
        "voice": voice or settings.default_voice,
        "response_format": response_format,
        "stream_format": "sse",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.rstrip("\r")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].lstrip()
                audio = _decode_sse_data_payload(payload)
                if audio:
                    yield audio


async def synthesize_full(
    text: str,
    *,
    model: str | None = None,
    voice: str | None = None,
    response_format: str = "mp3",
) -> bytes:
    """Buffer full speech audio (mp3/wav/pcm per response_format)."""
    parts: list[bytes] = []
    async for piece in synthesize_stream(
        text, model=model, voice=voice, response_format=response_format
    ):
        parts.append(piece)
    if not parts:
        logger.warning("TTS returned no decodable audio chunks; response may use a different SSE shape")
    return b"".join(parts)
