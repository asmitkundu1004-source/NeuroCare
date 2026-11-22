"""NeuroCare server: the closed affective loop behind one WebSocket.

Per turn:  sense -> safety gate -> fuse -> route persona -> AGEM recall -> generate -> speak
Every stage reports what it decided, so the interface can show the person why Aura
answered the way it did.

Run:  uvicorn neurocare.server:app --port 8000      (from backend/)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import llm, tts
from .affect import AffectState, Channel, fuse
from .config import BACKEND_DIR, settings
from .memory import MemoryStore
from .personas import PERSONAS, route, system_prompt
from .safety import MEDICAL_INSTRUCTION, SAFETY_INSTRUCTION, STICKY_TURNS, SafetyGate
from .sense import get_sense

log = logging.getLogger("neurocare")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MIN_WORDS_TO_REMEMBER = 4
MAX_INPUT_CHARS = 2000


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    app.state.sense = await asyncio.to_thread(get_sense)
    app.state.gate = SafetyGate()
    app.state.store = MemoryStore()
    await asyncio.to_thread(tts.engine)
    status = await llm.available()
    log.info("Aura-Sense loaded, %d memories, LLM %s %s, ready in %.1fs",
             app.state.store.stats()["memories"], settings.llm_model,
             "reachable" if status["ok"] else "not reachable (reflective fallback)", time.perf_counter() - t0)
    yield


app = FastAPI(title="NeuroCare", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- REST


class SenseIn(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


@app.get("/api/health")
async def health():
    sense = app.state.sense
    return {
        "status": "ok",
        "llm": await llm.available(),
        "tts": tts.engine() is not None,
        "sense": {"name": sense.meta["name"], "version": sense.meta["version"], "metrics": sense.meta.get("metrics")},
        "store": app.state.store.stats(),
        "region": settings.region,
    }


@app.get("/api/personas")
async def personas():
    return [{"id": p.id, "name": p.name, "framework": p.framework} for p in PERSONAS.values()]


@app.post("/api/sense")
async def sense_text(body: SenseIn):
    t0 = time.perf_counter()
    r = (await asyncio.to_thread(app.state.sense.read, body.text))[0]
    a = r.affect
    return {"affect": {**a.__dict__, "top": a.top(5)}, "lexicon": app.state.gate.scan(body.text),
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.get("/api/memories")
async def memories(limit: int = 200):
    return app.state.store.list_memories(limit)


@app.delete("/api/memories/{memory_id}")
async def forget(memory_id: int):
    if not app.state.store.forget(memory_id):
        raise HTTPException(404, "No such memory")
    return {"ok": True}


@app.get("/api/timeline")
async def timeline(days: int = 30):
    return app.state.store.timeline(max(1, min(days, 365)))


@app.get("/api/export")
async def export():
    return JSONResponse(app.state.store.export_all(),
                        headers={"Content-Disposition": 'attachment; filename="neurocare-export.json"'})


@app.post("/api/wipe")
async def wipe():
    app.state.store.wipe()
    return {"ok": True}


# --------------------------------------------------------------------------- conversation


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prev: AffectState | None = None
    sticky: int = 0
    persona: str = "empathetic"
    region: str = settings.region
    task: asyncio.Task | None = None


async def run_turn(ws: WebSocket, s: Session, msg: dict):
    text = str(msg.get("text", "")).strip()[:MAX_INPUT_CHARS]
    if not text:
        return
    store, gate, sense = app.state.store, app.state.gate, app.state.sense
    t0 = time.perf_counter()

    reading = (await asyncio.to_thread(sense.read, text))[0]
    t_sense = time.perf_counter()
    safety = gate.assess(text, reading.affect.risk, sense.risk_elevated, sense.risk_crisis, s.sticky)
    s.sticky = STICKY_TURNS if safety.tier == 2 else max(0, s.sticky - 1)
    state = fuse(reading.affect, Channel.parse(msg.get("face")), Channel.parse(msg.get("voice")), s.prev, safety.tier)
    s.prev = state

    requested = str(msg.get("persona", "auto"))
    if requested == "auto":
        persona_id, why = route(state, text, s.persona)
    else:
        persona_id, why = (requested if requested in PERSONAS else "empathetic"), "chosen by you"
    s.persona = persona_id
    persona = PERSONAS[persona_id]

    turn = store.next_turn()
    recalled = [] if safety.tier == 2 else await asyncio.to_thread(store.recall, reading.embedding, state.valence, turn)
    t_recall = time.perf_counter()

    await ws.send_json({
        "type": "affect", "state": state.to_dict(),
        "persona": {"id": persona_id, "name": persona.name, "reason": why, "auto": requested == "auto"},
        "safety": {**safety.to_dict(), "resources": gate.resources_for(s.region) if safety.tier else []},
        "memories": recalled,
        "timing": {"sense_ms": round((t_sense - t0) * 1000, 1), "recall_ms": round((t_recall - t_sense) * 1000, 1)},
    })

    history = store.history(s.id, settings.context_turns * 2)
    store.log_turn(s.id, "user", text, state.valence, state.arousal, state.dominant, safety.tier, persona_id)
    if safety.tier < 2 and len(text.split()) >= MIN_WORDS_TO_REMEMBER:
        store.remember(text, reading.embedding, reading.affect.valence, reading.affect.arousal, state.dominant)

    reply, first_token, fallback = "", None, False

    async def emit_sentence(sentence: str):
        audio = await asyncio.to_thread(tts.synthesize, sentence)
        await ws.send_json({"type": "sentence", "text": sentence, "audio": audio})

    if safety.tier == 2:
        reply = gate.crisis_message(s.region)
        first_token = time.perf_counter()
        await ws.send_json({"type": "token", "text": reply})
        for para in reply.split("\n\n"):
            await emit_sentence(para.replace("\n- ", ". ").replace("\n", " "))
    else:
        extra = []
        if safety.tier == 1:
            extra.append(SAFETY_INSTRUCTION)
        if safety.medical:
            extra.append(MEDICAL_INSTRUCTION)
        messages = [{"role": "system", "content": system_prompt(persona, state, recalled, extra)},
                    *history, {"role": "user", "content": text}]
        buffer = ""
        try:
            if not (await llm.available())["ok"]:
                raise ConnectionError("language model not reachable")
            async for tok in llm.stream(messages, persona.temperature, persona.top_p):
                first_token = first_token or time.perf_counter()
                reply += tok
                buffer += tok
                await ws.send_json({"type": "token", "text": tok})
                ready, buffer = tts.split_ready(buffer)
                for sentence in ready:
                    await emit_sentence(sentence)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the model being down must not end the conversation
            if not reply:
                log.warning("LLM unavailable (%s), using reflective fallback", type(e).__name__)
                fallback = True
                reply = buffer = llm.reflect(state, persona_id)
                first_token = time.perf_counter()
                await ws.send_json({"type": "token", "text": reply})
        if buffer.strip():
            await emit_sentence(buffer.strip())

    done = time.perf_counter()
    store.log_turn(s.id, "assistant", reply, persona=persona_id)
    await ws.send_json({
        "type": "done", "text": reply, "fallback": fallback,
        "timing": {"first_token_ms": round(((first_token or done) - t0) * 1000, 1),
                   "total_ms": round((done - t0) * 1000, 1)},
    })


async def _guarded(ws: WebSocket, s: Session, msg: dict):
    try:
        await run_turn(ws, s, msg)
    except asyncio.CancelledError:
        pass
    except Exception:
        log.exception("turn failed")
        try:
            await ws.send_json({"type": "error", "message": "Something went wrong on this turn. Please try again."})
        except Exception:  # noqa: BLE001
            pass


@app.websocket("/ws")
async def conversation(ws: WebSocket):
    await ws.accept()
    s = Session()
    await ws.send_json({"type": "ready", "session": s.id})
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "hello":
                s.region = str(msg.get("region", s.region))[:8]
            elif kind == "chat":
                if s.task and not s.task.done():
                    s.task.cancel()
                s.task = asyncio.create_task(_guarded(ws, s, msg))
            elif kind == "interrupt":
                if s.task and not s.task.done():
                    s.task.cancel()
                await ws.send_json({"type": "interrupted"})
            elif kind == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        if s.task and not s.task.done():
            s.task.cancel()


# --------------------------------------------------------------------------- web app

WEB_DIST = Path(os.environ.get("NEUROCARE_WEB_DIST", str(BACKEND_DIR.parent / "web" / "dist")))

if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str):
        file = (WEB_DIST / path).resolve()
        if path and file.is_file() and WEB_DIST.resolve() in file.parents:
            return FileResponse(file)
        return FileResponse(WEB_DIST / "index.html")
