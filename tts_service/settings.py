"""App settings.

Architecture (mixed Mify + Volcengine):

- **Mify HTTP API** (`mify_base_url`): TTS and ASR use `Authorization: Bearer` and
  `X-Model-Provider-Id` + model strings. Mify routes to upstreams (e.g. xiaomi for speech,
  volcengine_maas for ASR) — you do *not* use Volcengine AK/SK for those calls.

- **Volcengine Python SDK** (env `VOLC_*`): Jimeng video (`visual.volcengineapi.com`) and
  optional IAM probe (`iam.volcengineapi.com`) use AK/SK directly — separate from Mify.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of `tts_service/` (works no matter where you run uvicorn from).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOTENV = _REPO_ROOT / ".env"
if _DOTENV.is_file():
    load_dotenv(_DOTENV, override=False, encoding="utf-8")


class Settings(BaseSettings):
    # Also list the path so pydantic reads the same file (declared fields only).
    model_config = SettingsConfigDict(
        env_file=str(_DOTENV) if _DOTENV.is_file() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Mify gateway (speech) ---
    mify_api_key: str
    mify_base_url: str = "http://model.mify.ai.srv"
    mify_provider_id: str = "xiaomi"
    default_model: str = "tts-multivoice-v1"
    # Must match a voice your Mify+xiaomi route exposes (see Mify console / docs).
    default_voice: str = "zh_male_guanlan_prem"

    # ASR via Mify; provider id selects Volcengine ASR *through* Mify (not direct Volc API)
    asr_provider_id: str = "volcengine_maas"
    default_asr_model: str = "volc.bigasr.auc_turbo"


settings = Settings()
