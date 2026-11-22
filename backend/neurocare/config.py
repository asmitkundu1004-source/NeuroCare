"""Runtime configuration. Every value can be overridden with an environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"NEUROCARE_{name}", default)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(BACKEND_DIR / "data"))))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", str(BACKEND_DIR / "models" / "aura-sense"))))
    lexicon_path: Path = field(default_factory=lambda: Path(_env("LEXICON", str(BACKEND_DIR / "models" / "safety_lexicon.json"))))
    threads: int = int(_env("THREADS", "4"))
    region: str = _env("REGION", "IN")

    llm_url: str = _env("LLM_URL", "http://localhost:11434")
    llm_model: str = _env("LLM_MODEL", "aura")
    llm_timeout: float = float(_env("LLM_TIMEOUT", "120"))
    context_turns: int = int(_env("CONTEXT_TURNS", "6"))

    tts_enabled: bool = _env("TTS", "1") == "1"
    tts_voice: str = _env("TTS_VOICE", "af_sarah")

    # Passphrase for at-rest encryption of memory text. When unset a random key is
    # generated on first run and stored next to the database (still local-only).
    passphrase: str | None = os.environ.get("NEUROCARE_PASSPHRASE")

    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in _env("CORS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()
    )


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
