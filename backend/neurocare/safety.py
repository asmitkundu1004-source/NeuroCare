"""Safety gate. Runs before the language model on every turn and can overrule it.

Tier 0  no concern                 -> normal generation
Tier 1  elevated (distress cues or risk classifier above its high-recall threshold)
                                   -> generation continues under a safety instruction;
                                      the UI offers support resources without alarm
Tier 2  crisis (explicit ideation, intent or self-harm language, or the classifier
        above its high-precision threshold together with a distress cue)
                                   -> the model is bypassed and a fixed, reviewed
                                      message with helplines is returned

The lexicon is the safety net for short, explicit messages; the classifier catches
indirect language the lexicon cannot enumerate. A crisis keeps the session at
tier >= 1 for the next few turns so one calmer message does not drop all support.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings

STICKY_TURNS = 4


@dataclass
class SafetyResult:
    tier: int
    reasons: list[str] = field(default_factory=list)
    medical: bool = False

    def to_dict(self) -> dict:
        return {"tier": self.tier, "reasons": self.reasons, "medical": self.medical}


def normalise(text: str) -> str:
    t = text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    return re.sub(r"\s+", " ", t).strip()


class SafetyGate:
    def __init__(self, path: Path = settings.lexicon_path):
        lex = json.loads(Path(path).read_text(encoding="utf-8"))

        def compile_all(key):
            return [re.compile(p, re.IGNORECASE) for p in lex[key]]

        self.exclusions = compile_all("exclusions")
        self.crisis = compile_all("crisis")
        self.distress = compile_all("distress")
        self.medical = compile_all("medical")
        self.resources: dict[str, list[dict]] = lex["resources"]

    def scan(self, text: str) -> dict[str, list[str]]:
        t = normalise(text)
        for rx in self.exclusions:
            t = rx.sub(" ", t)
        hits = {}
        for key in ("crisis", "distress", "medical"):
            found = [m.group(0) for rx in getattr(self, key) if (m := rx.search(t))]
            if found:
                hits[key] = found
        return hits

    def assess(self, text: str, risk_prob: float, elevated_at: float, crisis_at: float, sticky: int = 0) -> SafetyResult:
        hits = self.scan(text)
        reasons: list[str] = []
        tier = 0
        if "crisis" in hits:
            tier, reasons = 2, [f"explicit language: '{hits['crisis'][0]}'"]
        elif risk_prob >= crisis_at and "distress" in hits:
            tier, reasons = 2, [f"risk model {risk_prob:.2f} with distress cue '{hits['distress'][0]}'"]
        elif "distress" in hits:
            tier, reasons = 1, [f"distress cue: '{hits['distress'][0]}'"]
        elif risk_prob >= elevated_at:
            tier, reasons = 1, [f"risk model {risk_prob:.2f} >= {elevated_at:.2f}"]
        if tier < 1 and sticky > 0:
            tier, reasons = 1, ["recent crisis in this session"]
        return SafetyResult(tier=tier, reasons=reasons, medical="medical" in hits)

    def resources_for(self, region: str = settings.region) -> list[dict]:
        return self.resources.get(region, []) + self.resources.get("INTL", [])

    def crisis_message(self, region: str = settings.region) -> str:
        lines = "\n".join(f"- {r['name']}: {r['contact']}" for r in self.resources_for(region))
        return (
            "I'm really glad you told me, and I'm taking what you said seriously. "
            "You don't have to carry this by yourself, and you deserve support from a real person right now.\n\n"
            f"Please reach out to one of these. They are free and confidential:\n{lines}\n\n"
            "If you are in immediate danger, please call emergency services or go to the nearest hospital. "
            "If you can, tell someone you trust what's going on and ask them to stay with you. "
            "I'm still here, and you can keep talking to me while you reach out."
        )


SAFETY_INSTRUCTION = (
    "SAFETY: The person may be in distress. Respond with warmth and without judgement. "
    "Gently ask how they are coping and whether they are safe. Do not minimise or argue with their feelings. "
    "Mention, once and softly, that talking to someone they trust or a helpline can help. "
    "Never give instructions, methods or information that could be used for self-harm."
)

MEDICAL_INSTRUCTION = (
    "BOUNDARY: The person is asking for a diagnosis, a medication change or a dose. You are not a clinician. "
    "Do not name doses, diagnose, or advise starting, stopping or changing medication. Acknowledge the question "
    "kindly and encourage them to discuss it with their doctor or psychiatrist, then support how they feel about it."
)
