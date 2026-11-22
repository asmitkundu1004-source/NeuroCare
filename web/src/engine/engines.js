// Two interchangeable engines behind one event stream:
//   ServerEngine  the FastAPI backend (fine-tuned Aura LLM via Ollama, Kokoro voice)
//   DeviceEngine  everything in this tab: Aura-Sense in WASM, AGEM over an encrypted
//                 IndexedDB vault, and an optional WebGPU language model (WebLLM)
// Events: affect | token | sentence | done | error | status | interrupted

import {
  AuraSense, MEDICAL_INSTRUCTION, PERSONA, SAFETY_INSTRUCTION, STICKY_TURNS, assess, crisisMessage,
  fuse, rankMemories, reflect, resourcesFor, route, systemPrompt,
} from './core.js'
import { Vault } from './vault.js'

const API = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const MIN_WORDS_TO_REMEMBER = 4
const CONTEXT_TURNS = 6
const SENTENCE_END = /(?<=[.!?])\s+(?=[A-Z0-9"'])/

export const DEVICE_MODELS = [
  { id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 1B', size: '0.9 GB' },
  { id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC', label: 'Llama 3.2 3B', size: '2.3 GB' },
]

class Emitter {
  constructor() { this.subs = new Set() }
  on(fn) { this.subs.add(fn); return () => this.subs.delete(fn) }
  emit(e) { this.subs.forEach((fn) => fn(e)) }
}

function splitReady(buffer) {
  const parts = buffer.split(SENTENCE_END)
  if (parts.length === 1) return [[], buffer]
  return [parts.slice(0, -1).map((s) => s.trim()).filter(Boolean), parts.at(-1)]
}

// ---------------------------------------------------------------- server

function wsUrl() {
  if (API) return API.replace(/^http/, 'ws') + '/ws'
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`
}

export async function probeServer(timeout = 1500) {
  try {
    const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(timeout) })
    if (!r.ok) return null
    const j = await r.json()
    return j.status === 'ok' ? j : null
  } catch {
    return null
  }
}

export class ServerEngine extends Emitter {
  kind = 'server'

  constructor(health) {
    super()
    this.health = health
    this.region = 'IN'
  }

  async init() {
    await this.connect()
    return this
  }

  connect() {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl())
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'hello', region: this.region }))
        resolve()
      }
      ws.onerror = () => reject(new Error('Could not reach the NeuroCare server'))
      ws.onmessage = (m) => this.emit(JSON.parse(m.data))
      ws.onclose = () => {
        this.emit({ type: 'status', connected: false })
        clearTimeout(this.retry)
        this.retry = setTimeout(() => this.connect().then(() => this.emit({ type: 'status', connected: true })).catch(() => {}), 2500)
      }
      this.ws = ws
    })
  }

  status() {
    const h = this.health || {}
    return {
      mode: 'server', label: 'Local server',
      llm: h.llm?.ok ? `${h.llm.model} · fine-tuned Llama 3.2 3B` : 'Reflective responder (no LLM running)',
      llmReady: Boolean(h.llm?.ok), voice: h.tts ? 'Kokoro-82M' : 'Browser voice', encryption: h.store?.encryption,
    }
  }

  setRegion(r) {
    this.region = r
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'hello', region: r }))
  }

  send(msg) {
    if (this.ws?.readyState !== 1) {
      this.emit({ type: 'error', message: 'Reconnecting to the server. Try again in a moment.' })
      return
    }
    this.ws.send(JSON.stringify({ type: 'chat', ...msg }))
  }

  interrupt() {
    if (this.ws?.readyState === 1) this.ws.send(JSON.stringify({ type: 'interrupt' }))
  }

  json(path, init) { return fetch(`${API}${path}`, init).then((r) => r.json()) }
  memories() { return this.json('/api/memories') }
  forget(id) { return this.json(`/api/memories/${id}`, { method: 'DELETE' }) }
  timeline(days) { return this.json(`/api/timeline?days=${days}`) }
  exportAll() { return this.json('/api/export') }
  wipe() { return this.json('/api/wipe', { method: 'POST' }) }
  async stats() { return (await this.json('/api/health')).store }
  async sense(text) {
    const r = await this.json('/api/sense', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text }) })
    return { affect: r.affect, ms: r.ms }
  }

  close() {
    clearTimeout(this.retry)
    if (this.ws) {
      this.ws.onclose = null
      this.ws.close()
    }
  }
}

// ---------------------------------------------------------------- device

export class DeviceEngine extends Emitter {
  kind = 'device'

  constructor() {
    super()
    this.session = crypto.randomUUID().slice(0, 12)
    this.prev = null
    this.sticky = 0
    this.persona = 'empathetic'
    this.region = 'IN'
    this.llm = null
    this.llmName = null
    this.llmLoading = false
    this.generation = 0
  }

  async init(onProgress) {
    const [model, vault] = await Promise.all([AuraSense.load(onProgress), Vault.open()])
    this.model = model
    this.vault = vault
    const saved = localStorage.getItem('nc.deviceModel')
    if (saved && 'gpu' in navigator) this.loadLLM(saved).catch(() => localStorage.removeItem('nc.deviceModel'))
    return this
  }

  get meta() { return this.model.meta }

  status() {
    return {
      mode: 'device', label: 'On this device',
      llm: this.llm ? `${this.llmName} · WebGPU` : this.llmLoading ? `Loading ${this.llmName}…` : 'Reflective responder',
      llmReady: Boolean(this.llm), voice: 'Browser voice', encryption: 'AES-256-GCM (non-extractable browser key)',
      webgpu: 'gpu' in navigator,
    }
  }

  async loadLLM(modelId, onProgress) {
    if (!('gpu' in navigator)) throw new Error('This browser does not support WebGPU')
    const model = DEVICE_MODELS.find((m) => m.id === modelId)
    this.llmName = model?.label || modelId
    this.llmLoading = true
    this.emit({ type: 'status' })
    try {
      const { CreateMLCEngine } = await import('@mlc-ai/web-llm')
      this.llm = await CreateMLCEngine(modelId, {
        initProgressCallback: (p) => {
          onProgress?.(p)
          this.emit({ type: 'status', progress: p })
        },
      })
      localStorage.setItem('nc.deviceModel', modelId)
    } finally {
      this.llmLoading = false
      this.emit({ type: 'status' })
    }
  }

  async unloadLLM() {
    await this.llm?.unload?.()
    this.llm = null
    localStorage.removeItem('nc.deviceModel')
    this.emit({ type: 'status' })
  }

  setRegion(r) { this.region = r }

  interrupt() {
    this.generation += 1
    this.llm?.interruptGenerate?.()
    this.emit({ type: 'interrupted' })
  }

  async sense(text) {
    const r = await this.model.read(text)
    return { affect: r.affect, ms: r.ms }
  }

  send(msg) {
    const gen = ++this.generation
    this.turn(msg, gen).catch((err) => {
      console.error(err)
      if (gen === this.generation) this.emit({ type: 'error', message: 'Something went wrong on this turn. Please try again.' })
    })
  }

  async turn(msg, gen) {
    const text = String(msg.text || '').trim().slice(0, 2000)
    if (!text) return
    const alive = () => gen === this.generation
    const t0 = performance.now()
    const reading = await this.model.read(text)
    const tSense = performance.now()
    const m = this.meta
    const safety = assess(text, reading.affect.risk, m.risk.threshold_elevated, m.risk.threshold_crisis, this.sticky)
    this.sticky = safety.tier === 2 ? STICKY_TURNS : Math.max(0, this.sticky - 1)
    const state = fuse(reading.affect, msg.face, msg.voice, this.prev, safety.tier)
    this.prev = state

    const auto = !msg.persona || msg.persona === 'auto'
    const [personaId, reason] = auto ? route(state, text, this.persona) : [PERSONA[msg.persona] ? msg.persona : 'empathetic', 'chosen by you']
    this.persona = personaId
    const persona = PERSONA[personaId]

    const turn = await this.vault.nextTurn()
    let memories = []
    if (safety.tier < 2) {
      memories = rankMemories(reading.embedding, this.vault.memories(), state.valence, turn).map((c) => ({
        id: c.memory.id, text: c.memory.text, similarity: +c.similarity.toFixed(3), score: +c.score.toFixed(3),
        valence: +c.memory.valence.toFixed(3), dominant: c.memory.dominant, created: c.memory.created / 1000,
        admitted: !c.gated, gated: c.gated,
      }))
      await this.vault.markSurfaced(memories.filter((x) => x.admitted).map((x) => x.id), turn)
    }
    if (!alive()) return
    this.emit({
      type: 'affect', state, memories,
      persona: { id: personaId, name: persona.name, reason, auto },
      safety: { ...safety, resources: safety.tier ? resourcesFor(this.region) : [] },
      timing: { sense_ms: +(tSense - t0).toFixed(1), recall_ms: +(performance.now() - tSense).toFixed(1) },
    })

    const history = await this.vault.history(this.session, CONTEXT_TURNS * 2)
    await this.vault.logTurn({ session: this.session, role: 'user', text, valence: state.valence, arousal: state.arousal, dominant: state.dominant, risk_tier: safety.tier, persona: personaId })
    if (safety.tier < 2 && text.split(/\s+/).length >= MIN_WORDS_TO_REMEMBER) {
      await this.vault.remember(text, reading.embedding, reading.affect.valence, reading.affect.arousal, state.dominant)
    }

    let reply = ''
    let firstToken = null
    let fallback = false
    const emitWhole = (whole) => {
      firstToken = performance.now()
      this.emit({ type: 'token', text: whole })
      for (const s of whole.split('\n\n')) this.emit({ type: 'sentence', text: s.replace(/\n- /g, '. ').replace(/\n/g, ' ') })
    }

    if (safety.tier === 2) {
      reply = crisisMessage(this.region)
      emitWhole(reply)
    } else if (this.llm) {
      const extra = []
      if (safety.tier === 1) extra.push(SAFETY_INSTRUCTION)
      if (safety.medical) extra.push(MEDICAL_INSTRUCTION)
      const messages = [{ role: 'system', content: systemPrompt(persona, state, memories, extra) }, ...history, { role: 'user', content: text }]
      let buffer = ''
      const stream = await this.llm.chat.completions.create({ messages, stream: true, temperature: persona.temperature, top_p: persona.top_p, max_tokens: 320 })
      for await (const chunk of stream) {
        if (!alive()) return
        const tok = chunk.choices[0]?.delta?.content
        if (!tok) continue
        firstToken ||= performance.now()
        reply += tok
        buffer += tok
        this.emit({ type: 'token', text: tok })
        const [ready, rest] = splitReady(buffer)
        buffer = rest
        ready.forEach((s) => this.emit({ type: 'sentence', text: s }))
      }
      if (buffer.trim()) this.emit({ type: 'sentence', text: buffer.trim() })
    } else {
      fallback = true
      reply = reflect(state, personaId)
      await new Promise((r) => setTimeout(r, 450))
      if (!alive()) return
      emitWhole(reply)
    }

    if (!alive()) return
    const done = performance.now()
    await this.vault.logTurn({ session: this.session, role: 'assistant', text: reply, persona: personaId })
    this.emit({ type: 'done', text: reply, fallback, timing: { first_token_ms: +((firstToken || done) - t0).toFixed(1), total_ms: +(done - t0).toFixed(1) } })
  }

  memories() {
    return Promise.resolve(
      this.vault.memories().map(({ embedding: _e, ...m }) => ({ ...m, created: m.created / 1000 })).sort((a, b) => b.created - a.created),
    )
  }
  forget(id) { return this.vault.forget(id) }
  timeline(days) { return this.vault.timeline(days) }
  exportAll() { return this.vault.exportAll() }
  wipe() { return this.vault.wipe() }
  stats() { return this.vault.stats() }
  close() { this.generation += 1 }
}

// ---------------------------------------------------------------- selection

export async function createEngine(preference = 'auto', onProgress) {
  if (preference !== 'device') {
    onProgress?.('Looking for a local NeuroCare server')
    const health = await probeServer()
    if (health) {
      try {
        return await new ServerEngine(health).init()
      } catch { /* fall through to the device engine */ }
    }
    if (preference === 'server') throw new Error('No NeuroCare server found. Start it with: uvicorn neurocare.server:app')
  }
  return new DeviceEngine().init(onProgress)
}
