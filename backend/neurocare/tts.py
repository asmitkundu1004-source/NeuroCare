"""Optional on-device speech with Kokoro-82M.

Synthesis runs per sentence while the model is still generating the rest, so the first
words are heard after one sentence rather than after the whole reply. When pykokoro is
not installed the browser's own speech synthesis is used instead.
"""

from __future__ import annotations

import base64
import io
import logging
import re
import wave

import numpy as np

from .config import settings

log = logging.getLogger("neurocare.tts")
_pipe = None
_tried = False

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def engine():
    global _pipe, _tried
    if not _tried and settings.tts_enabled:
        _tried = True
        try:
            from pykokoro import KokoroPipeline, PipelineConfig

            _pipe = KokoroPipeline(PipelineConfig(provider="auto", voice=settings.tts_voice))
            log.info("Kokoro TTS ready (%s)", settings.tts_voice)
        except Exception as e:  # noqa: BLE001
            log.info("Kokoro TTS unavailable (%s); the browser will speak instead", type(e).__name__)
    return _pipe


def synthesize(text: str) -> str | None:
    """Return a base64 WAV for one sentence, or None."""
    pipe = engine()
    if pipe is None or not text.strip():
        return None
    try:
        res = pipe.run(text)
        pcm = (np.clip(res.audio, -1, 1) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(res.sample_rate)
            w.writeframes(pcm.tobytes())
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # noqa: BLE001
        log.warning("TTS failed: %s", e)
        return None


def split_ready(buffer: str) -> tuple[list[str], str]:
    """Split complete sentences off the front of a streaming buffer."""
    parts = SENTENCE_END.split(buffer)
    if len(parts) == 1:
        return [], buffer
    return [p.strip() for p in parts[:-1] if p.strip()], parts[-1]
