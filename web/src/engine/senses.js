// Face and voice, read in the browser. Only small numeric summaries leave this file:
// video frames and raw audio are never stored or sent anywhere.

import { BASE, clamp } from './core.js'

// ---------------------------------------------------------------- shared math

function erf(x) {
  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const s = Math.sign(x)
  const t = 1 / (1 + 0.3275911 * Math.abs(x))
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)
  return s * y
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2))
const f32 = (b64) => new Float32Array(Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer)

function dense(layer, x, act) {
  const [out, inp] = layer.shape
  const y = new Float32Array(out)
  for (let o = 0; o < out; o++) {
    let s = layer.b[o]
    const row = o * inp
    for (let i = 0; i < inp; i++) s += layer.W[row + i] * x[i]
    y[o] = act ? gelu(s) : s
  }
  return y
}

function softmax(z) {
  const m = Math.max(...z)
  const e = Array.from(z, (v) => Math.exp(v - m))
  const sum = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / sum)
}

// ---------------------------------------------------------------- Aura-Face

export class FaceSense {
  static async create(video, circumplex) {
    const vision = await import('@mediapipe/tasks-vision')
    const files = await vision.FilesetResolver.forVisionTasks(`${BASE}mediapipe`)
    const make = (delegate) => vision.FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetPath: `${BASE}models/face_landmarker.task`, delegate },
      runningMode: 'VIDEO', outputFaceBlendshapes: true, numFaces: 1,
    })
    const landmarker = await make('GPU').catch(() => make('CPU'))
    const spec = await fetch(`${BASE}models/aura-face.json`).then((r) => r.json())
    return new FaceSense(video, landmarker, spec, circumplex)
  }

  constructor(video, landmarker, spec, circumplex) {
    this.video = video
    this.landmarker = landmarker
    this.spec = spec
    this.mean = f32(spec.mean)
    this.std = f32(spec.std)
    this.members = spec.members.map((layers) => layers.map((l) => ({ shape: l.shape, W: f32(l.W), b: f32(l.b) })))
    this.anchor = spec.labels.map((l) => circumplex?.[spec.to_goemotions[l]] || [0, 0])
    this.probs = null
    this.blend = null
    this.present = false
    this.running = false
    this.lastTs = -1
  }

  features(scores) {
    const { single, pairs } = this.spec
    const x = [
      ...single.map((i) => scores[i]),
      ...pairs.map(([l, r]) => (scores[l] + scores[r]) / 2),
      ...pairs.map(([l, r]) => Math.abs(scores[l] - scores[r])),
    ]
    return Float32Array.from(x, (v, i) => (v - this.mean[i]) / this.std[i])
  }

  classify(scores) {
    const x = this.features(scores)
    const acc = new Array(this.spec.labels.length).fill(0)
    for (const layers of this.members) {
      const h1 = dense(layers[0], x, true)
      const h2 = dense(layers[1], h1, true)
      softmax(dense(layers[2], h2, false)).forEach((p, i) => { acc[i] += p / this.members.length })
    }
    return acc
  }

  start(onUpdate, fps = 8) {
    this.running = true
    const order = this.spec.blendshapes
    let last = 0
    const loop = (now) => {
      if (!this.running) return
      if (now - last > 1000 / fps && this.video.readyState >= 2) {
        last = now
        const ts = Math.max(this.lastTs + 1, Math.round(now))
        this.lastTs = ts
        const res = this.landmarker.detectForVideo(this.video, ts)
        const cats = res.faceBlendshapes?.[0]?.categories
        this.present = Boolean(cats)
        if (cats) {
          const byName = Object.fromEntries(cats.map((c) => [c.categoryName, c.score]))
          const p = this.classify(order.map((n) => byName[n] ?? 0))
          this.probs = this.probs ? this.probs.map((q, i) => 0.7 * q + 0.3 * p[i]) : p
          this.blend = byName
        }
        onUpdate?.(this.reading())
      }
      this.raf = requestAnimationFrame(loop)
    }
    this.raf = requestAnimationFrame(loop)
  }

  reading() {
    if (!this.present || !this.probs) return { present: false, confidence: 0 }
    const labels = this.spec.labels
    let v = 0, a = 0
    this.probs.forEach((p, i) => { v += p * this.anchor[i][0]; a += p * this.anchor[i][1] })
    const top = this.probs.indexOf(Math.max(...this.probs))
    const chance = 1 / labels.length
    const confidence = clamp((this.probs[top] - chance) / (1 - chance), 0, 1)
    return {
      present: true, label: labels[top], probs: Object.fromEntries(labels.map((l, i) => [l, this.probs[i]])),
      valence: v, arousal: a, confidence, blend: this.blend,
    }
  }

  stop() {
    this.running = false
    cancelAnimationFrame(this.raf)
  }

  close() {
    this.stop()
    this.landmarker.close?.()
  }
}

// ---------------------------------------------------------------- voice in: transcription + prosody

const Recognition = typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition)

function pitch(buf, rate) {
  // autocorrelation over 70-400 Hz
  let best = 0, bestLag = 0
  const minLag = Math.floor(rate / 400), maxLag = Math.floor(rate / 70)
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0
    for (let i = 0; i + lag < buf.length; i += 2) s += buf[i] * buf[i + lag]
    if (s > best) { best = s; bestLag = lag }
  }
  return bestLag ? rate / bestLag : 0
}

const BASELINE_KEY = 'nc.prosody.baseline'

export class Ear {
  static supported = Boolean(Recognition)

  constructor() {
    this.onDevice = false
    this.level = 0
  }

  async start({ onInterim, lang = 'en-US' } = {}) {
    this.text = ''
    this.interim = ''
    this.frames = []
    this.t0 = performance.now()
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
    this.ctx = new AudioContext()
    const src = this.ctx.createMediaStreamSource(this.stream)
    this.analyser = this.ctx.createAnalyser()
    this.analyser.fftSize = 2048
    src.connect(this.analyser)
    const buf = new Float32Array(this.analyser.fftSize)
    this.timer = setInterval(() => {
      this.analyser.getFloatTimeDomainData(buf)
      let e = 0
      for (const x of buf) e += x * x
      const rms = Math.sqrt(e / buf.length)
      this.level = Math.min(1, rms * 8)
      this.frames.push({ rms, f0: rms > 0.01 ? pitch(buf, this.ctx.sampleRate) : 0 })
    }, 60)

    if (Recognition) {
      const r = new Recognition()
      r.lang = lang
      r.continuous = true
      r.interimResults = true
      if ('processLocally' in r) {
        try {
          const ok = await Recognition.available?.({ langs: [lang], processLocally: true })
          if (ok === 'available') { r.processLocally = true; this.onDevice = true }
        } catch { /* older engines fall back to the browser's own speech service */ }
      }
      r.onresult = (ev) => {
        let interim = ''
        for (let i = ev.resultIndex; i < ev.results.length; i++) {
          const res = ev.results[i]
          if (res.isFinal) this.text = `${this.text} ${res[0].transcript}`.trim()
          else interim += res[0].transcript
        }
        this.interim = interim
        onInterim?.(`${this.text} ${interim}`.trim())
      }
      r.onerror = (e) => { if (e.error === 'not-allowed') this.listening = false }
      r.onend = () => {
        if (this.listening) {
          try { r.start() } catch { /* already running */ }
        }
      }
      this.recognition = r
      this.listening = true
      r.start()
    }
  }

  prosody(words) {
    const voiced = this.frames.filter((f) => f.rms > 0.01)
    const secs = (performance.now() - this.t0) / 1000
    if (voiced.length < 5) return null
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length
    const energy = 20 * Math.log10(mean(voiced.map((f) => f.rms)) + 1e-6)
    const f0s = voiced.map((f) => f.f0).filter((f) => f > 0)
    const f0sd = f0s.length > 3 ? Math.sqrt(mean(f0s.map((f) => (f - mean(f0s)) ** 2))) : 0
    const rate = words / Math.max(secs, 0.5)
    const cur = { energy, f0sd, rate }
    let base = null
    try { base = JSON.parse(localStorage.getItem(BASELINE_KEY)) } catch { /* ignore a corrupt baseline */ }
    let arousal, confidence
    if (base && base.n >= 2) {
      // relative to this person's own speaking baseline
      const z = (k, s) => (cur[k] - base[k]) / s
      arousal = Math.tanh(0.5 * z('energy', 6) + 0.3 * z('f0sd', 25) + 0.2 * z('rate', 1))
      confidence = 0.55
    } else {
      arousal = Math.tanh((energy + 30) / 12 + (rate - 2.5) / 2)
      confidence = 0.25
    }
    const n = (base?.n || 0) + 1
    const blend = (k) => (base ? base[k] + (cur[k] - base[k]) / Math.min(n, 20) : cur[k])
    localStorage.setItem(BASELINE_KEY, JSON.stringify({ n, energy: blend('energy'), f0sd: blend('f0sd'), rate: blend('rate') }))
    return { valence: null, arousal: clamp(arousal), confidence, energy, f0sd, rate }
  }

  async stop() {
    this.listening = false
    clearInterval(this.timer)
    await new Promise((r) => setTimeout(r, 350))
    try { this.recognition?.stop() } catch { /* not running */ }
    this.stream?.getTracks().forEach((t) => t.stop())
    this.ctx?.close()
    this.level = 0
    const text = `${this.text} ${this.interim}`.trim()
    return { text, prosody: this.prosody(text.split(/\s+/).filter(Boolean).length) }
  }

  abort() {
    this.listening = false
    clearInterval(this.timer)
    try { this.recognition?.abort() } catch { /* not running */ }
    this.stream?.getTracks().forEach((t) => t.stop())
    this.ctx?.close()
    this.level = 0
  }
}

// ---------------------------------------------------------------- voice out

function pickVoice() {
  const voices = speechSynthesis.getVoices().filter((v) => v.lang.startsWith('en'))
  const prefer = [/natural/i, /aria|jenny|sonia|libby/i, /samantha|karen|moira/i, /google uk english female/i, /female/i]
  for (const rx of prefer) {
    const v = voices.find((x) => rx.test(x.name))
    if (v) return v
  }
  return voices[0] || null
}

export class Speaker {
  constructor() {
    this.queue = []
    this.playing = false
    this.level = 0
    this.enabled = true
    this.listeners = new Set()
  }

  on(fn) {
    this.listeners.add(fn)
    return () => this.listeners.delete(fn)
  }

  emit(speaking) {
    this.listeners.forEach((fn) => fn(speaking))
  }

  say(text, audioB64) {
    if (!this.enabled || !text.trim()) return
    this.queue.push({ text, audioB64 })
    if (!this.playing) this.next()
  }

  async next() {
    const item = this.queue.shift()
    if (!item) {
      this.playing = false
      this.level = 0
      this.emit(false)
      return
    }
    if (!this.playing) {
      this.playing = true
      this.emit(true)
    }
    try {
      if (item.audioB64) await this.playWav(item.audioB64)
      else await this.synth(item.text)
    } catch { /* skip a sentence that fails to play */ }
    if (this.playing) this.next()
  }

  async playWav(b64) {
    this.ctx ||= new AudioContext()
    if (this.ctx.state === 'suspended') await this.ctx.resume()
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    const buf = await this.ctx.decodeAudioData(bytes.buffer)
    const src = this.ctx.createBufferSource()
    const an = this.ctx.createAnalyser()
    an.fftSize = 512
    src.buffer = buf
    src.connect(an).connect(this.ctx.destination)
    const data = new Uint8Array(an.fftSize)
    this.source = src
    return new Promise((resolve) => {
      const tick = () => {
        if (this.source !== src) return resolve()
        an.getByteTimeDomainData(data)
        let e = 0
        for (const v of data) e += ((v - 128) / 128) ** 2
        this.level = Math.min(1, Math.sqrt(e / data.length) * 4)
        this.raf = requestAnimationFrame(tick)
      }
      src.onended = () => {
        cancelAnimationFrame(this.raf)
        this.level = 0
        resolve()
      }
      src.start()
      tick()
    })
  }

  synth(text) {
    return new Promise((resolve) => {
      if (!('speechSynthesis' in window)) return resolve()
      const u = new SpeechSynthesisUtterance(text)
      u.voice = pickVoice()
      u.rate = 0.98
      let pulse
      u.onstart = () => { pulse = setInterval(() => { this.level = 0.35 + Math.random() * 0.4 }, 90) }
      u.onboundary = () => { this.level = 0.9 }
      u.onend = u.onerror = () => {
        clearInterval(pulse)
        this.level = 0
        resolve()
      }
      speechSynthesis.speak(u)
    })
  }

  stop() {
    this.queue = []
    const was = this.playing
    this.playing = false
    try { this.source?.stop() } catch { /* already stopped */ }
    this.source = null
    if ('speechSynthesis' in window) speechSynthesis.cancel()
    this.level = 0
    if (was) this.emit(false)
  }
}
