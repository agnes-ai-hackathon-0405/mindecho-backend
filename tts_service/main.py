"""HTTP API: TTS (text → audio), ASR (audio → text) via Mify, and Jimeng video via Volcengine."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Annotated
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from tts_service.asr_text import extract_transcript_text
from tts_service.mify_errors import mify_upstream_http_exception
from tts_service.mify_asr import transcribe as asr_transcribe
from tts_service.mify_asr import transcribe_base64
from tts_service.mify_tts import synthesize_full, synthesize_stream
from tts_service.volc_errors import http_exception_from_volc_sdk
from tts_service.volc_iam import iam_probe
from tts_service.settings import settings
from tts_service.volc_jimeng_video import generate_video_pro

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mify Voice + Jimeng Video API", version="1.0.0")


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    model: str | None = Field(None, description="Override model (default from env)")
    voice: str | None = Field(None, description="Override voice id")
    response_format: str = Field("mp3", description="mp3, wav, or pcm")


class TranscribeJsonRequest(BaseModel):
    """Same shape as Mify: base64 under audio.data, wrapped for convenience."""

    audio_base64: str = Field(
        ..., min_length=1, description="Base64-encoded audio (pcm/wav/mp3/ogg opus)"
    )
    model: str | None = Field(None, description="Override ASR model")
    conversation_id: str | None = None
    model_request_id: str | None = None
    user_id: str | None = None


class VoiceToVideoResponse(BaseModel):
    """ASR transcript + [即梦视频生成 3.0 Pro](https://www.volcengine.com/docs/85621/1777001?lang=zh) result URL."""

    transcript: str = Field(..., description="Text from speech recognition")
    video_url: str = Field(
        ..., description="HTTPS URL of the generated video (from Volcengine)"
    )
    duration_sec: int
    aspect_ratio: str
    asr_raw: dict | None = Field(None, description="Raw ASR JSON if include_asr=true")


class VideoFromTextRequest(BaseModel):
    """Text prompt → Volcengine 即梦 3.0 Pro video (no Mify)."""

    text: str = Field(
        ...,
        min_length=1,
        description="Video prompt (positive_prompt); UTF-8 Chinese or any language supported by 即梦",
    )
    duration_sec: int = Field(5, description="5 or 10 seconds")
    aspect_ratio: str = "16:9"
    seed: int = -1
    req_key: str | None = Field(None, description="Override VOLC_JIMENG_REQ_KEY")
    video_timeout: float = Field(1800.0, description="Poll timeout seconds")


class VideoFromTextResponse(BaseModel):
    prompt: str
    video_url: str
    duration_sec: int
    aspect_ratio: str


class FetchVideoUrlRequest(BaseModel):
    """Download a Jimeng result URL through this server (browsers abroad often cannot resolve *.aigc-cloud.com)."""

    url: str = Field(..., min_length=12, description="HTTPS URL returned by /video/from-text")


# Hosts Jimeng / Volcengine use for temporary video URLs (SSRF allowlist).
_ALLOWED_VIDEO_HOST_SUFFIXES = (
    ".aigc-cloud.com",
    ".volces.com",
    ".volcengine.com",
    ".bytecdn.cn",
    ".byteimg.com",
)


def _is_allowed_jimeng_cdn_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
        if p.scheme not in ("http", "https"):
            return False
        h = (p.hostname or "").lower()
        return any(h.endswith(suf) for suf in _ALLOWED_VIDEO_HOST_SUFFIXES)
    except Exception:
        return False


def _ensure_volc_env() -> None:
    """Jimeng Visual API expects VOLC_ACCESSKEY / VOLC_SECRETKEY (see volc_jimeng_video)."""
    ak = (os.environ.get("VOLC_ACCESSKEY") or os.environ.get("VOLC_ACCESS_KEY") or "").strip()
    sk = (os.environ.get("VOLC_SECRETKEY") or os.environ.get("VOLC_SECRET_KEY") or "").strip()
    if not ak or not sk:
        raise HTTPException(
            status_code=503,
            detail="Set VOLC_ACCESSKEY and VOLC_SECRETKEY for Jimeng video (Volcengine console keys).",
        )
    os.environ["VOLC_ACCESSKEY"] = ak
    os.environ["VOLC_SECRETKEY"] = sk


async def _jimeng_generate_video_url(
    prompt: str,
    *,
    duration_sec: int,
    aspect_ratio: str,
    seed: int,
    req_key: str | None,
    video_timeout: float,
) -> str:
    """Volcengine Jimeng only (`generate_video_pro`)."""
    if duration_sec not in (5, 10):
        raise HTTPException(status_code=400, detail="duration_sec must be 5 or 10")
    _ensure_volc_env()
    p = prompt.strip()
    if not p:
        raise HTTPException(status_code=400, detail="Empty prompt")

    def _run() -> tuple[str, dict, dict]:
        return generate_video_pro(
            p,
            req_key=req_key,
            aspect_ratio=aspect_ratio,
            duration_sec=duration_sec,
            seed=seed,
            timeout=video_timeout,
        )

    try:
        video_url, _, _ = await asyncio.to_thread(_run)
        return video_url
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.warning("Jimeng / Visual API error: %s", e)
        raise http_exception_from_volc_sdk(e) from e


@app.post("/video/from-text", response_model=VideoFromTextResponse)
async def video_from_text(body: VideoFromTextRequest) -> VideoFromTextResponse:
    """
    **Text → video** via Volcengine 即梦 3.0 Pro only (no Mify, no audio).

    Requires `VOLC_ACCESSKEY` and `VOLC_SECRETKEY`. Same backend as `/video/from-voice` after transcription.
    """
    video_url = await _jimeng_generate_video_url(
        body.text,
        duration_sec=body.duration_sec,
        aspect_ratio=body.aspect_ratio,
        seed=body.seed,
        req_key=body.req_key,
        video_timeout=body.video_timeout,
    )
    return VideoFromTextResponse(
        prompt=body.text.strip(),
        video_url=video_url,
        duration_sec=body.duration_sec,
        aspect_ratio=body.aspect_ratio,
    )


@app.post("/video/fetch-url")
async def video_fetch_url(body: FetchVideoUrlRequest) -> Response:
    """
    Download the MP4 through this service and return bytes. Use when the browser shows **host not found**
    for `*.aigc-cloud.com` (common outside mainland China DNS). This host must be able to resolve the CDN.

    Short clips are buffered in memory (typical Jimeng 5–10s outputs).
    """
    if not _is_allowed_jimeng_cdn_url(body.url):
        raise HTTPException(
            status_code=400,
            detail=f"URL host must match one of: {_ALLOWED_VIDEO_HOST_SUFFIXES}",
        )
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "video/mp4,video/*,*/*",
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600.0),
        follow_redirects=True,
    ) as client:
        try:
            r = await client.get(body.url, headers=headers)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=(e.response.text or "")[:4000],
            ) from e
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Could not download video URL",
                    "error": str(e),
                    "hint": (
                        "CDN hostnames often resolve only on mainland China networks. "
                        "Run the API on a CN host/VPN, or download with curl from a CN machine."
                    ),
                },
            ) from e
    media = r.headers.get("content-type", "video/mp4")
    return Response(
        content=r.content,
        media_type=media,
        headers={
            "Content-Disposition": 'attachment; filename="jimeng_video.mp4"',
        },
    )


@app.get("/volc/iam/probe")
async def volc_iam_probe(
    strict: Annotated[
        bool,
        Query(
            description="If true, return 403 when no IAM call succeeds (keys invalid or no IAM permission).",
        ),
    ] = False,
) -> dict:
    """
    Exercise **IAM** `IamService` with the same AK/SK as Jimeng (`set_ak` / `set_sk`, per
    [volc-sdk-python](https://github.com/volcengine/volc-sdk-python)). Read-only: `ListAccessKeys`, `GetUser`.

    Use this to confirm credentials work against `iam.volcengineapi.com`. Failing here does not rule out
    Visual-only keys; conversely, success here does not grant 即梦 video until product/IAM allows it.
    """
    _ensure_volc_env()
    ak = os.environ["VOLC_ACCESSKEY"]
    sk = os.environ["VOLC_SECRETKEY"]

    def _run() -> dict:
        return iam_probe(ak, sk)

    result = await asyncio.to_thread(_run)
    if strict and not result.get("ok"):
        raise HTTPException(
            status_code=403,
            detail=result,
        )
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config/mify")
def mify_config_snapshot() -> dict:
    """
    Effective Mify **model / voice / provider** IDs (no secrets). If you see
    `Not supported model string`, change these via `.env` (`DEFAULT_MODEL`, `DEFAULT_VOICE`,
    `DEFAULT_ASR_MODEL`, `MIFY_PROVIDER_ID`, `ASR_PROVIDER_ID`) and restart.
    """
    return {
        "mify_base_url": settings.mify_base_url,
        "tts": {
            "X-Model-Provider-Id": settings.mify_provider_id,
            "default_model": settings.default_model,
            "default_voice": settings.default_voice,
        },
        "asr": {
            "X-Model-Provider-Id": settings.asr_provider_id,
            "default_asr_model": settings.default_asr_model,
        },
        "env_keys": {
            "tts": "DEFAULT_MODEL, DEFAULT_VOICE, MIFY_PROVIDER_ID, MIFY_BASE_URL",
            "asr": "DEFAULT_ASR_MODEL, ASR_PROVIDER_ID",
        },
    }


@app.post("/tts")
async def tts_json(body: TtsRequest) -> Response:
    """Synthesize speech and return a single audio file (buffered)."""
    try:
        audio = await synthesize_full(
            body.text,
            model=body.model,
            voice=body.voice,
            response_format=body.response_format,
        )
    except httpx.HTTPStatusError as e:  # type: ignore[name-defined]
        raise mify_upstream_http_exception(e) from e
    except Exception as e:
        logger.exception("TTS request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e

    if not audio:
        raise HTTPException(
            status_code=502,
            detail="Empty audio from upstream; check API key, network, or SSE format.",
        )

    media = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(body.response_format.lower(), "application/octet-stream")

    return Response(content=audio, media_type=media)


@app.post("/transcribe")
async def transcribe_upload(
    file: Annotated[UploadFile, File(description="Audio: pcm, wav, mp3, ogg opus")],
    model: Annotated[str | None, Form()] = None,
    conversation_id: Annotated[str | None, Form()] = None,
    x_conversation_id: Annotated[str | None, Header(alias="X-Conversation-Id")] = None,
    x_model_request_id: Annotated[
        str | None, Header(alias="X-Model-Request-Id")
    ] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> dict:
    """Voice recognition: upload an audio file; returns upstream JSON (text fields depend on API)."""
    cid = x_conversation_id if x_conversation_id is not None else conversation_id
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return await asr_transcribe(
            audio_bytes,
            model=model,
            conversation_id=cid,
            model_request_id=x_model_request_id,
            user_id=x_user_id,
        )
    except httpx.HTTPStatusError as e:
        raise mify_upstream_http_exception(e) from e
    except Exception as e:
        logger.exception("ASR request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/transcribe/json")
async def transcribe_json_body(body: TranscribeJsonRequest) -> dict:
    """Same as /transcribe but with JSON body `{\"audio_base64\": \"...\"}` (curl-style base64)."""
    try:
        return await transcribe_base64(
            body.audio_base64,
            model=body.model,
            conversation_id=body.conversation_id,
            model_request_id=body.model_request_id,
            user_id=body.user_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except httpx.HTTPStatusError as e:
        raise mify_upstream_http_exception(e) from e
    except Exception as e:
        logger.exception("ASR request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.post("/video/from-voice", response_model=VoiceToVideoResponse)
async def video_from_voice(
    file: Annotated[
        UploadFile, File(description="Voice audio (same formats as /transcribe)")
    ],
    duration_sec: Annotated[int, Form()] = 5,
    aspect_ratio: Annotated[str, Form()] = "16:9",
    seed: Annotated[int, Form()] = -1,
    req_key: Annotated[str | None, Form()] = None,
    prompt_prefix: Annotated[str, Form()] = "",
    prompt_suffix: Annotated[str, Form()] = "",
    include_asr: Annotated[bool, Form()] = False,
    model: Annotated[str | None, Form()] = None,
    conversation_id: Annotated[str | None, Form()] = None,
    x_conversation_id: Annotated[str | None, Header(alias="X-Conversation-Id")] = None,
    x_model_request_id: Annotated[
        str | None, Header(alias="X-Model-Request-Id")
    ] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    video_timeout: Annotated[float, Form()] = 1800.0,
) -> VoiceToVideoResponse:
    """
    Transcribe uploaded audio (same path as `/transcribe`), then call **Volcengine 即梦 视频生成 3.0 Pro**
    (`CVSync2AsyncSubmitTask` → poll `CVSync2AsyncGetResult`) with the transcript as the video prompt.

    Requires `VOLC_ACCESSKEY` and `VOLC_SECRETKEY`. Optional `VOLC_JIMENG_REQ_KEY` (default `jimeng_ti2v_v30_pro`).
    """
    cid = x_conversation_id if x_conversation_id is not None else conversation_id
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        asr_raw = await asr_transcribe(
            audio_bytes,
            model=model,
            conversation_id=cid,
            model_request_id=x_model_request_id,
            user_id=x_user_id,
        )
    except httpx.HTTPStatusError as e:
        raise mify_upstream_http_exception(e) from e
    except Exception as e:
        logger.exception("ASR request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e

    transcript = extract_transcript_text(asr_raw)
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not extract transcript text from ASR response",
                "hint": "Check ASR JSON shape or pass text via another route",
                "asr_raw": asr_raw if include_asr else None,
            },
        )

    prompt = f"{prompt_prefix}{transcript}{prompt_suffix}".strip()
    if not prompt:
        prompt = transcript

    video_url = await _jimeng_generate_video_url(
        prompt,
        duration_sec=duration_sec,
        aspect_ratio=aspect_ratio,
        seed=seed,
        req_key=req_key,
        video_timeout=video_timeout,
    )

    return VoiceToVideoResponse(
        transcript=transcript,
        video_url=video_url,
        duration_sec=duration_sec,
        aspect_ratio=aspect_ratio,
        asr_raw=asr_raw if include_asr else None,
    )


@app.post("/tts/stream")
async def tts_stream(
    body: TtsRequest,
) -> StreamingResponse:
    """Stream audio chunks as they arrive from Mify (same codec as response_format)."""
    media = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }.get(body.response_format.lower(), "application/octet-stream")

    async def gen():
        async for chunk in synthesize_stream(
            body.text,
            model=body.model,
            voice=body.voice,
            response_format=body.response_format,
        ):
            yield chunk

    return StreamingResponse(gen(), media_type=media)


@app.get("/tts")
async def tts_get(
    text: Annotated[str, Query(min_length=1)],
    voice: str | None = None,
    model: str | None = None,
    fmt: str = Query("mp3", alias="format"),
) -> Response:
    """Convenience GET: /tts?text=...&format=mp3"""
    try:
        audio = await synthesize_full(
            text, model=model, voice=voice, response_format=fmt
        )
    except httpx.HTTPStatusError as e:  # type: ignore[name-defined]
        raise mify_upstream_http_exception(e) from e
    except Exception as e:
        logger.exception("TTS request failed")
        raise HTTPException(status_code=502, detail=str(e)) from e

    if not audio:
        raise HTTPException(status_code=502, detail="Empty audio from upstream.")

    media = {"mp3": "audio/mpeg", "wav": "audio/wav", "pcm": "audio/pcm"}.get(
        fmt.lower(), "application/octet-stream"
    )
    return Response(content=audio, media_type=media)
