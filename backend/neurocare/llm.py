"""Streaming client for the local language model.

Default is Ollama serving the fine-tuned Aura model (Llama-3.2-3B, LoRA, Q4_K_M GGUF).
Any OpenAI-compatible server (llama.cpp server, LM Studio, vLLM) also works:
    NEUROCARE_LLM_API=openai NEUROCARE_LLM_URL=http://localhost:8080
If no model is reachable the server degrades to a small reflective responder instead
of failing, so sensing, safety and memory keep working.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import AsyncIterator

import httpx

from .affect import AffectState
from .config import settings

API = os.environ.get("NEUROCARE_LLM_API", "ollama")


_cache: tuple[float, dict] | None = None


async def available(max_age: float = 15.0) -> dict:
    """Reachability of the model, cached briefly so a missing server costs nothing per turn."""
    global _cache
    now = time.monotonic()
    if _cache and now - _cache[0] < max_age:
        return _cache[1]
    _cache = (now, await _probe())
    return _cache[1]


async def _probe() -> dict:
    try:
        async with httpx.AsyncClient(timeout=2.5) as c:
            if API == "ollama":
                r = await c.get(f"{settings.llm_url}/api/tags")
                names = [m["name"] for m in r.json().get("models", [])]
                ok = any(n.split(":")[0].lower() == settings.llm_model.lower() for n in names)
                return {"ok": ok, "api": API, "model": settings.llm_model, "installed": names}
            r = await c.get(f"{settings.llm_url}/v1/models")
            return {"ok": r.status_code == 200, "api": API, "model": settings.llm_model}
    except Exception as e:  # noqa: BLE001 - any failure means "not available"
        return {"ok": False, "api": API, "model": settings.llm_model, "error": type(e).__name__}


async def stream(messages: list[dict], temperature: float, top_p: float) -> AsyncIterator[str]:
    timeout = httpx.Timeout(settings.llm_timeout, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as c:
        if API == "ollama":
            body = {"model": settings.llm_model, "messages": messages, "stream": True,
                    "options": {"temperature": temperature, "top_p": top_p, "num_ctx": 4096, "repeat_penalty": 1.1}}
            async with c.stream("POST", f"{settings.llm_url}/api/chat", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if tok := chunk.get("message", {}).get("content"):
                        yield tok
                    if chunk.get("done"):
                        return
        else:
            body = {"model": settings.llm_model, "messages": messages, "stream": True,
                    "temperature": temperature, "top_p": top_p}
            async with c.stream("POST", f"{settings.llm_url}/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:") or line.strip() == "data: [DONE]":
                        continue
                    delta = json.loads(line[5:])["choices"][0].get("delta", {})
                    if tok := delta.get("content"):
                        yield tok


# --------------------------------------------------------------------------- fallback

# The reflective responder speaks when no language model is reachable. It keys off the
# dominant emotion (more reliable than the arousal estimate) and the chosen stance.
MOOD_OF = {
    "anxious": {"fear", "nervousness", "confusion"},
    "low": {"sadness", "grief", "disappointment", "remorse", "embarrassment"},
    "angry": {"anger", "annoyance", "disapproval", "disgust"},
    "bright": {"joy", "amusement", "excitement", "gratitude", "love", "optimism", "pride", "relief", "admiration", "approval"},
}
OPENERS = {
    "anxious": ["That sounds like a lot of pressure to be carrying.", "It makes sense to feel on edge with that hanging over you.",
                "I can hear how much this is weighing on you."],
    "low": ["That sounds really heavy.", "I'm sorry it feels this way right now.", "That sounds hard, and it makes sense that it hurts."],
    "angry": ["That sounds genuinely frustrating.", "I can hear how much that got to you.", "It makes sense to feel angry about that."],
    "bright": ["I love hearing that.", "That sounds genuinely good.", "That's lovely. I'm glad you told me."],
    "settled": ["I'm here and listening.", "Thank you for telling me.", "I'm glad you're talking about it."],
}
FOLLOW = {
    "empathetic": ["What part of it is sitting with you the most?", "How has it been feeling, day to day?"],
    "zen": ["Before anything else, try one slow breath with me, letting the exhale run longer than the inhale.",
            "Can you notice your feet on the floor for a moment, and let your shoulders drop a little?"],
    "strategist": ["What is the thought that keeps coming back, word for word?",
                   "If a friend said that about themselves, what would you point out to them?"],
    "motivator": ["What is one tiny thing you could do in the next ten minutes?", "What would a small, doable first step look like today?"],
    "caregiver": ["Have you had a chance to eat, drink some water or rest today?", "What would be one kind thing you could do for yourself tonight?"],
    "stoic": ["Which part of this is in your hands, and which part isn't?", "What still matters to you here, even with all of this going on?"],
}


TRAP = "It sounds like your mind might be jumping to the hardest version of this."
_last_reflection = ""


def reflect(state: AffectState, persona_id: str) -> str:
    global _last_reflection
    mood = next((k for k, v in MOOD_OF.items() if state.dominant in v), "settled")
    # A cheerful label on an unpleasant turn ("I keep thinking I'm going to fail" reads as
    # optimism to GoEmotions) must not get a cheerful reply: trust the valence head instead.
    if mood == "bright" and state.quadrant not in ("bright", "calm"):
        mood = {"tense": "anxious", "low": "low"}.get(state.quadrant, "settled")
    use_trap = (state.distortion_type and persona_id == "strategist" and mood != "bright"
                and not _last_reflection.startswith(TRAP))
    reply = _last_reflection
    for _ in range(8):
        opener = TRAP if use_trap else random.choice(OPENERS[mood])
        follow = "What made it feel that way?" if mood == "bright" else random.choice(FOLLOW.get(persona_id, FOLLOW["empathetic"]))
        reply = f"{opener} {follow}"
        if reply != _last_reflection:
            break
    _last_reflection = reply
    return reply
