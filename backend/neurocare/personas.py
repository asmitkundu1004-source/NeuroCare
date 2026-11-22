"""Therapeutic personas and the affect-driven router that picks between them.

Each persona is a therapeutic stance rather than a character: a framework, a register
and sampling settings. In "auto" mode the router reads the fused affect state every
turn and chooses the stance that best fits it, and says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .affect import AffectState


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    framework: str
    temperature: float
    top_p: float
    prompt: str


BASE = (
    "You are Aura, a warm companion inside NeuroCare, a private app that runs on the person's own device. "
    "You are not a therapist and you never claim to be one. Speak the way a thoughtful person talks out loud: "
    "short paragraphs, no lists, no headings, no emoji, no markdown. Usually two to four sentences. "
    "Ask at most one question at a time. Never diagnose and never give medication advice."
)

PERSONAS: dict[str, Persona] = {p.id: p for p in [
    Persona("empathetic", "The Listener", "Person-centred, reflective listening", 0.7, 0.9,
            "Your stance is person-centred. Reflect back what you hear, name the feeling you notice, and let the "
            "person feel understood before anything else. Do not offer advice unless they ask for it."),
    Persona("zen", "The Anchor", "Mindfulness and grounding", 0.4, 0.85,
            "Your stance is grounding. Slow the pace down. When someone is anxious or overwhelmed, guide one small "
            "body-based practice, such as a slow exhale longer than the inhale, or noticing five things they can see. "
            "Keep sentences short and calm."),
    Persona("strategist", "The Reframer", "Cognitive behavioural therapy", 0.5, 0.9,
            "Your stance is cognitive behavioural. Help the person look at the thought separately from the feeling. "
            "If you notice a thinking trap such as catastrophising, mind reading or all-or-nothing thinking, name it "
            "gently and ask one question that tests it against the evidence."),
    Persona("motivator", "The Coach", "Behavioural activation", 0.75, 0.92,
            "Your stance is behavioural activation. Be encouraging and practical. Help the person find one small, "
            "concrete step they could take in the next ten minutes, and make it feel doable rather than demanding."),
    Persona("caregiver", "The Caretaker", "Self-compassion and basic needs", 0.65, 0.9,
            "Your stance is self-compassion. Gently check on the basics: sleep, food, water, rest, company. Speak to "
            "the person the way a kind friend would, and remind them they are allowed to go easy on themselves."),
    Persona("stoic", "The Philosopher", "Stoic resilience and meaning", 0.6, 0.9,
            "Your stance draws on Stoic practice and the search for meaning. Help the person separate what is in "
            "their control from what is not, and find what still matters to them here. Quiet, steady, never preachy."),
]}

PROCRASTINATION = re.compile(
    r"\b(procrastinat\w*|can'?t (get )?start\w*|no motivation|unmotivated|lazy|putting (it|things) off|stuck on)\b", re.I)
DEPLETION = re.compile(
    r"\b(exhausted|tired|drained|burn(t|ed) out|can'?t sleep|insomnia|haven'?t (eaten|slept)|no energy|sick)\b", re.I)


def route(state: AffectState, text: str, current: str) -> tuple[str, str]:
    """Pick a persona for this turn from the fused affect state. Returns (id, reason)."""
    d, v, a = state.dominant, state.valence, state.arousal
    if state.risk_tier >= 1:
        if a > 0.15:
            return "zen", "high distress with agitation: grounding first"
        return "empathetic", "high distress: being heard comes first"
    if d in ("fear", "nervousness") or (state.quadrant == "tense" and a > 0.2):
        return "zen", f"{d} and high arousal: slow things down"
    if state.distortion_type and v < 0.05:
        return "strategist", f"possible {state.distortion_type.lower()}: look at the thought"
    if d in ("anger", "annoyance", "disapproval", "disgust"):
        return "stoic", f"{d}: separate what can and cannot be controlled"
    if DEPLETION.search(text):
        return "caregiver", "signs of depletion: check the basics"
    if PROCRASTINATION.search(text):
        return "motivator", "feeling stuck: one small step"
    if d in ("grief", "sadness", "disappointment", "remorse", "embarrassment") or state.quadrant == "low":
        return "empathetic", f"{d}: reflect and validate"
    return (current if current in PERSONAS else "empathetic"), "steady state: keep the current stance"


def system_prompt(persona: Persona, state: AffectState, memories: list[dict], extra: list[str]) -> str:
    feeling = ", ".join(k for k, p in state.emotions if p >= 0.15 and k != "neutral") or "calm or neutral"
    parts = [BASE, persona.prompt,
             f"What the person seems to be feeling right now (read on-device from their words, and voice and face "
             f"if they enabled them): {feeling}; overall mood is {state.quadrant}. Let this shape your tone. Do not "
             f"mention scores or that you are reading their emotions unless they ask."]
    admitted = [m for m in memories if m.get("admitted")]
    if admitted:
        notes = "\n".join(f"- {m['text']}" for m in admitted)
        parts.append("Things the person told you in earlier conversations that relate to this moment:\n"
                     f"{notes}\nBring one up only if it genuinely helps, and never list them.")
    parts.extend(extra)
    return "\n\n".join(parts)
