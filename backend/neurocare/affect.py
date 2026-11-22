"""Multimodal affect fusion.

Text affect comes from Aura-Sense. Face (Aura-Face on blendshapes) and voice (prosody)
are computed in the browser and arrive as tiny numeric summaries; frames and audio
never reach the server. All three channels live in the same valence/arousal space:
Aura-Face projects its class probabilities through the circumplex Aura-Sense learned,
so the modalities are directly comparable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .sense import TextAffect

# Per-dimension trust in each channel, before confidence weighting. Text carries
# meaning and dominates valence; prosody tracks arousal well but valence poorly
# (Scherer, 2003); facial action units sit in between.
PRIORS = {
    "valence": {"text": 0.60, "face": 0.30, "voice": 0.10},
    "arousal": {"text": 0.40, "face": 0.25, "voice": 0.35},
}


@dataclass
class Channel:
    valence: float | None = None
    arousal: float | None = None
    confidence: float = 0.0
    label: str | None = None

    @classmethod
    def parse(cls, raw: dict | None) -> "Channel | None":
        if not isinstance(raw, dict):
            return None

        def num(k):
            v = raw.get(k)
            return float(np.clip(v, -1, 1)) if isinstance(v, (int, float)) else None

        conf = raw.get("confidence", 0)
        conf = float(np.clip(conf, 0, 1)) if isinstance(conf, (int, float)) else 0.0
        label = str(raw["label"])[:24] if raw.get("label") else None
        return cls(num("valence"), num("arousal"), conf, label)


@dataclass
class AffectState:
    valence: float
    arousal: float
    quadrant: str
    dominant: str
    emotions: list[tuple[str, float]]
    risk: float
    risk_tier: int
    distortion: float
    distortion_type: str | None
    channels: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def quadrant(v: float, a: float) -> str:
    if abs(v) < 0.08 and abs(a) < 0.08:
        return "settled"
    if v >= 0:
        return "bright" if a >= 0 else "calm"
    return "tense" if a >= 0 else "low"


def fuse(text: TextAffect, face: Channel | None, voice: Channel | None, prev: AffectState | None,
         risk_tier: int, smoothing: float = 0.35) -> AffectState:
    """Confidence-weighted late fusion, then exponential smoothing across turns.

    Smoothing is skipped whenever the safety layer has raised a tier, so an acute
    change is never averaged away by a calmer history.
    """
    chans = {"text": Channel(text.valence, text.arousal, 1.0), "face": face or Channel(), "voice": voice or Channel()}

    def combine(dim: str) -> float:
        num = den = 0.0
        for name, ch in chans.items():
            val = getattr(ch, dim)
            if val is None or ch.confidence <= 0:
                continue
            w = PRIORS[dim][name] * ch.confidence
            num, den = num + w * val, den + w
        return num / den if den else 0.0

    v, a = combine("valence"), combine("arousal")
    if prev is not None and risk_tier == 0:
        v = smoothing * prev.valence + (1 - smoothing) * v
        a = smoothing * prev.arousal + (1 - smoothing) * a
    top = text.top(3)
    non_neutral = [k for k, _ in top if k != "neutral"]
    dominant = non_neutral[0] if top[0][0] == "neutral" and top[0][1] < 0.6 and non_neutral else top[0][0]
    return AffectState(
        valence=round(float(np.clip(v, -1, 1)), 4), arousal=round(float(np.clip(a, -1, 1)), 4),
        quadrant=quadrant(v, a), dominant=dominant, emotions=[(k, round(p, 3)) for k, p in top],
        risk=text.risk, risk_tier=risk_tier, distortion=text.distortion, distortion_type=text.distortion_type,
        channels={k: asdict(c) for k, c in chans.items()},
    )
