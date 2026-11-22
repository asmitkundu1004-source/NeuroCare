// Browser port of the NeuroCare loop: tokenizer, Aura-Sense, safety gate, fusion,
// AGEM memory ranking and the persona router. Mirrors backend/neurocare/*.py so the
// zero-server build behaves like the local server.

import lexicon from '../data/safety_lexicon.json'

export const BASE = import.meta.env.BASE_URL

// ---------------------------------------------------------------- tokenizer (BERT WordPiece)

const isPunct = (ch) => {
  const c = ch.codePointAt(0)
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) return true
  return /\p{P}/u.test(ch)
}
const isCJK = (c) =>
  (c >= 0x4e00 && c <= 0x9fff) || (c >= 0x3400 && c <= 0x4dbf) || (c >= 0x20000 && c <= 0x2a6df) ||
  (c >= 0xf900 && c <= 0xfaff) || (c >= 0x2f800 && c <= 0x2fa1f)

export class WordPiece {
  constructor(json) {
    this.vocab = new Map(Object.entries(json.model.vocab))
    this.unk = this.vocab.get('[UNK]')
    this.cls = this.vocab.get('[CLS]')
    this.sep = this.vocab.get('[SEP]')
  }

  basic(text) {
    let t = ''
    for (const ch of text) {
      const c = ch.codePointAt(0)
      if (c === 0 || c === 0xfffd || (/\p{Cc}/u.test(ch) && !/\s/.test(ch))) continue
      if (isCJK(c)) t += ` ${ch} `
      else t += /\s/.test(ch) ? ' ' : ch
    }
    t = t.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '')
    const out = []
    for (const word of t.split(' ')) {
      if (!word) continue
      let cur = ''
      for (const ch of word) {
        if (isPunct(ch)) {
          if (cur) out.push(cur)
          out.push(ch)
          cur = ''
        } else cur += ch
      }
      if (cur) out.push(cur)
    }
    return out
  }

  wordpiece(word) {
    const chars = [...word]
    if (chars.length > 100) return [this.unk]
    const ids = []
    let start = 0
    while (start < chars.length) {
      let end = chars.length
      let found = null
      while (start < end) {
        const piece = (start > 0 ? '##' : '') + chars.slice(start, end).join('')
        if (this.vocab.has(piece)) { found = this.vocab.get(piece); break }
        end -= 1
      }
      if (found === null) return [this.unk]
      ids.push(found)
      start = end
    }
    return ids
  }

  encode(text, maxLen = 128) {
    const ids = [this.cls]
    outer: for (const w of this.basic(text)) {
      for (const id of this.wordpiece(w)) {
        if (ids.length >= maxLen - 1) break outer
        ids.push(id)
      }
    }
    ids.push(this.sep)
    return ids
  }
}

// ---------------------------------------------------------------- Aura-Sense

const sigmoid = (z) => 1 / (1 + Math.exp(-z))
function softmax1(a, b) {
  const m = Math.max(a, b)
  const ea = Math.exp(a - m), eb = Math.exp(b - m)
  return eb / (ea + eb)
}

export const topK = (obj, k) => Object.entries(obj).sort((a, b) => b[1] - a[1]).slice(0, k)

export class AuraSense {
  static async load(onProgress) {
    const ort = await import('onnxruntime-web/wasm')
    ort.env.wasm.numThreads = 1
    const root = `${BASE}models/aura-sense/`
    onProgress?.('Loading Aura-Sense')
    const [meta, tok, buf] = await Promise.all([
      fetch(root + 'meta.json').then((r) => r.json()),
      fetch(root + 'tokenizer.json').then((r) => r.json()),
      fetch(root + 'model.onnx').then((r) => {
        if (!r.ok) throw new Error(`model.onnx: HTTP ${r.status}`)
        return r.arrayBuffer()
      }),
    ])
    const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' })
    return new AuraSense(ort, session, meta, new WordPiece(tok))
  }

  constructor(ort, session, meta, tokenizer) {
    this.ort = ort
    this.session = session
    this.meta = meta
    this.tokenizer = tokenizer
    this.labels = meta.emotion.labels
    this.thresholds = meta.emotion.thresholds
    this.circumplex = meta.circumplex
    this.queue = Promise.resolve()
  }

  // Inference calls are serialised; one WASM session must not run concurrently.
  read(text) {
    const run = this.queue.then(() => this._read(text))
    this.queue = run.catch(() => {})
    return run
  }

  async _read(text) {
    const t0 = performance.now()
    const ids = this.tokenizer.encode(text, this.meta.max_length || 128)
    const n = ids.length
    const feeds = {
      input_ids: new this.ort.Tensor('int64', BigInt64Array.from(ids, (x) => BigInt(x)), [1, n]),
      attention_mask: new this.ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
    }
    const out = await this.session.run(feeds)
    const logits = out.emotion_logits.data
    const emotions = {}
    this.labels.forEach((l, i) => { emotions[l] = sigmoid(logits[i]) })
    const active = this.labels.filter((l, i) => emotions[l] >= this.thresholds[i])
    const ekman = {}
    for (const [k, members] of Object.entries(this.meta.emotion.ekman)) ekman[k] = Math.max(...members.map((m) => emotions[m]))
    const r = out.risk_logits.data, d = out.distortion_logits.data
    const distortion = softmax1(d[0], d[1])
    let distortionType = null
    if (distortion >= this.meta.distortion.threshold) {
      const t = Array.from(out.distortion_type_logits.data)
      distortionType = this.meta.distortion.types[t.indexOf(Math.max(...t))]
    }
    return {
      embedding: Float32Array.from(out.embedding.data),
      affect: {
        emotions, active: active.length ? active : [topK(emotions, 1)[0][0]], ekman,
        valence: out.va.data[0], arousal: out.va.data[1],
        risk: softmax1(r[0], r[1]), distortion, distortion_type: distortionType,
      },
      ms: performance.now() - t0,
    }
  }
}

// ---------------------------------------------------------------- safety gate

const compile = (key) => lexicon[key].map((p) => new RegExp(p, 'i'))
const RX = { exclusions: lexicon.exclusions.map((p) => new RegExp(p, 'gi')), crisis: compile('crisis'), distress: compile('distress'), medical: compile('medical') }
export const STICKY_TURNS = 4

export function scan(text) {
  let t = text.toLowerCase().replace(/[‘’`]/g, "'").replace(/\s+/g, ' ').trim()
  for (const rx of RX.exclusions) t = t.replace(rx, ' ')
  const hits = {}
  for (const key of ['crisis', 'distress', 'medical']) {
    const found = RX[key].map((rx) => t.match(rx)?.[0]).filter(Boolean)
    if (found.length) hits[key] = found
  }
  return hits
}

export function assess(text, risk, elevatedAt, crisisAt, sticky = 0) {
  const hits = scan(text)
  let tier = 0
  let reasons = []
  if (hits.crisis) { tier = 2; reasons = [`explicit language: '${hits.crisis[0]}'`] }
  else if (risk >= crisisAt && hits.distress) { tier = 2; reasons = [`risk model ${risk.toFixed(2)} with distress cue '${hits.distress[0]}'`] }
  else if (hits.distress) { tier = 1; reasons = [`distress cue: '${hits.distress[0]}'`] }
  else if (risk >= elevatedAt) { tier = 1; reasons = [`risk model ${risk.toFixed(2)} ≥ ${elevatedAt.toFixed(2)}`] }
  if (tier < 1 && sticky > 0) { tier = 1; reasons = ['recent crisis in this session'] }
  return { tier, reasons, medical: Boolean(hits.medical) }
}

export const resourcesFor = (region) => [...(lexicon.resources[region] || []), ...lexicon.resources.INTL]
export const REGIONS = Object.keys(lexicon.resources).filter((r) => r !== 'INTL')

export function crisisMessage(region) {
  const lines = resourcesFor(region).map((r) => `- ${r.name}: ${r.contact}`).join('\n')
  return "I'm really glad you told me, and I'm taking what you said seriously. You don't have to carry this by yourself, " +
    'and you deserve support from a real person right now.\n\n' +
    `Please reach out to one of these. They are free and confidential:\n${lines}\n\n` +
    'If you are in immediate danger, please call emergency services or go to the nearest hospital. ' +
    "If you can, tell someone you trust what's going on and ask them to stay with you. " +
    "I'm still here, and you can keep talking to me while you reach out."
}

// ---------------------------------------------------------------- fusion

const PRIORS = {
  valence: { text: 0.6, face: 0.3, voice: 0.1 },
  arousal: { text: 0.4, face: 0.25, voice: 0.35 },
}

export function quadrant(v, a) {
  if (Math.abs(v) < 0.08 && Math.abs(a) < 0.08) return 'settled'
  if (v >= 0) return a >= 0 ? 'bright' : 'calm'
  return a >= 0 ? 'tense' : 'low'
}

export const clamp = (x, lo = -1, hi = 1) => Math.min(hi, Math.max(lo, x))

export function fuse(text, face, voice, prev, tier, smoothing = 0.35) {
  const ch = { text: { valence: text.valence, arousal: text.arousal, confidence: 1 }, face, voice }
  const combine = (dim) => {
    let num = 0, den = 0
    for (const [name, c] of Object.entries(ch)) {
      if (!c || c[dim] == null || !(c.confidence > 0)) continue
      const w = PRIORS[dim][name] * clamp(c.confidence, 0, 1)
      num += w * c[dim]
      den += w
    }
    return den ? num / den : 0
  }
  let v = combine('valence'), a = combine('arousal')
  if (prev && tier === 0) {
    v = smoothing * prev.valence + (1 - smoothing) * v
    a = smoothing * prev.arousal + (1 - smoothing) * a
  }
  const top = topK(text.emotions, 3)
  const nonNeutral = top.filter(([k]) => k !== 'neutral')
  const dominant = top[0][0] === 'neutral' && top[0][1] < 0.6 && nonNeutral.length ? nonNeutral[0][0] : top[0][0]
  return {
    valence: clamp(v), arousal: clamp(a), quadrant: quadrant(v, a), dominant,
    emotions: top.map(([k, p]) => [k, +p.toFixed(3)]), risk: text.risk, risk_tier: tier,
    distortion: text.distortion, distortion_type: text.distortion_type,
    channels: { text: ch.text, face: face || null, voice: voice || null },
  }
}

// ---------------------------------------------------------------- AGEM

export const AGEM = {
  k: 3, tau: 0.38, tauPainful: 0.55, painfulValence: -0.12, betaSalience: 0.25,
  gammaRecency: 0.05, halfLifeDays: 21, refractoryTurns: 3, refractoryPenalty: 0.6,
}

const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s }

export function rankMemories(query, memories, queryValence, turn, now = Date.now(), p = AGEM) {
  const personLow = queryValence <= p.painfulValence
  const scored = memories.map((m) => {
    const sim = dot(query, m.embedding)
    const sal = Math.min(1, Math.hypot(m.valence, Math.max(0, m.arousal)))
    const ageDays = (now - m.created) / 86400000
    const rec = Math.exp((-Math.LN2 * ageDays) / p.halfLifeDays)
    const refractory = turn - (m.lastTurn ?? -1e6) < p.refractoryTurns
    let score = sim * (1 + p.betaSalience * sal) + p.gammaRecency * rec
    if (refractory) score -= p.refractoryPenalty
    let gated = null
    if (sim < p.tau) gated = 'not relevant'
    else if (refractory) gated = 'recently surfaced'
    else if (m.valence <= p.painfulValence && !personLow && sim < p.tauPainful) gated = 'painful and only loosely related'
    return { memory: m, score, similarity: sim, gated }
  }).sort((a, b) => b.score - a.score)
  const admitted = scored.filter((c) => !c.gated).slice(0, p.k)
  const held = scored.filter((c) => c.gated && c.similarity >= p.tau * 0.8).slice(0, 3)
  return [...admitted, ...held]
}

// ---------------------------------------------------------------- personas

export const PERSONAS = [
  { id: 'empathetic', name: 'The Listener', framework: 'Person-centred reflection', temperature: 0.7, top_p: 0.9, hue: 22,
    blurb: 'Reflects what it hears and names the feeling, before anything else.',
    prompt: 'Your stance is person-centred. Reflect back what you hear, name the feeling you notice, and let the person feel understood before anything else. Do not offer advice unless they ask for it.' },
  { id: 'zen', name: 'The Anchor', framework: 'Mindfulness and grounding', temperature: 0.4, top_p: 0.85, hue: 160,
    blurb: 'Slows everything down and guides one small, body-based practice.',
    prompt: 'Your stance is grounding. Slow the pace down. When someone is anxious or overwhelmed, guide one small body-based practice, such as a slow exhale longer than the inhale, or noticing five things they can see. Keep sentences short and calm.' },
  { id: 'strategist', name: 'The Reframer', framework: 'Cognitive behavioural therapy', temperature: 0.5, top_p: 0.9, hue: 205,
    blurb: 'Separates the thought from the feeling and tests it gently against evidence.',
    prompt: 'Your stance is cognitive behavioural. Help the person look at the thought separately from the feeling. If you notice a thinking trap such as catastrophising, mind reading or all-or-nothing thinking, name it gently and ask one question that tests it against the evidence.' },
  { id: 'motivator', name: 'The Coach', framework: 'Behavioural activation', temperature: 0.75, top_p: 0.92, hue: 38,
    blurb: 'Finds one small, concrete step you could take in the next ten minutes.',
    prompt: 'Your stance is behavioural activation. Be encouraging and practical. Help the person find one small, concrete step they could take in the next ten minutes, and make it feel doable rather than demanding.' },
  { id: 'caregiver', name: 'The Caretaker', framework: 'Self-compassion and basic needs', temperature: 0.65, top_p: 0.9, hue: 335,
    blurb: 'Checks on sleep, food, water and rest, the way a kind friend would.',
    prompt: 'Your stance is self-compassion. Gently check on the basics: sleep, food, water, rest, company. Speak to the person the way a kind friend would, and remind them they are allowed to go easy on themselves.' },
  { id: 'stoic', name: 'The Philosopher', framework: 'Stoic resilience and meaning', temperature: 0.6, top_p: 0.9, hue: 250,
    blurb: 'Separates what is in your control from what is not, and finds what still matters.',
    prompt: 'Your stance draws on Stoic practice and the search for meaning. Help the person separate what is in their control from what is not, and find what still matters to them here. Quiet, steady, never preachy.' },
]
export const PERSONA = Object.fromEntries(PERSONAS.map((p) => [p.id, p]))

const PROCRASTINATION = /\b(procrastinat\w*|can'?t (get )?start\w*|no motivation|unmotivated|lazy|putting (it|things) off|stuck on)\b/i
const DEPLETION = /\b(exhausted|tired|drained|burn(t|ed) out|can'?t sleep|insomnia|haven'?t (eaten|slept)|no energy|sick)\b/i

export function route(state, text, current) {
  const { dominant: d, valence: v, arousal: a } = state
  if (state.risk_tier >= 1) return a > 0.15 ? ['zen', 'high distress with agitation: grounding first'] : ['empathetic', 'high distress: being heard comes first']
  if (d === 'fear' || d === 'nervousness' || (state.quadrant === 'tense' && a > 0.2)) return ['zen', `${d} and high arousal: slow things down`]
  if (state.distortion_type && v < 0.05) return ['strategist', `possible ${state.distortion_type.toLowerCase()}: look at the thought`]
  if (['anger', 'annoyance', 'disapproval', 'disgust'].includes(d)) return ['stoic', `${d}: separate what can and cannot be controlled`]
  if (DEPLETION.test(text)) return ['caregiver', 'signs of depletion: check the basics']
  if (PROCRASTINATION.test(text)) return ['motivator', 'feeling stuck: one small step']
  if (['grief', 'sadness', 'disappointment', 'remorse', 'embarrassment'].includes(d) || state.quadrant === 'low') return ['empathetic', `${d}: reflect and validate`]
  return [PERSONA[current] ? current : 'empathetic', 'steady state: keep the current stance']
}

const BASE_PROMPT = "You are Aura, a warm companion inside NeuroCare, a private app that runs on the person's own device. " +
  'You are not a therapist and you never claim to be one. Speak the way a thoughtful person talks out loud: ' +
  'short paragraphs, no lists, no headings, no emoji, no markdown. Usually two to four sentences. ' +
  'Ask at most one question at a time. Never diagnose and never give medication advice.'

export const SAFETY_INSTRUCTION = 'SAFETY: The person may be in distress. Respond with warmth and without judgement. Gently ask how they are coping and whether they are safe. Do not minimise or argue with their feelings. Mention, once and softly, that talking to someone they trust or a helpline can help. Never give instructions, methods or information that could be used for self-harm.'
export const MEDICAL_INSTRUCTION = 'BOUNDARY: The person is asking for a diagnosis, a medication change or a dose. You are not a clinician. Do not name doses, diagnose, or advise starting, stopping or changing medication. Acknowledge the question kindly and encourage them to discuss it with their doctor or psychiatrist, then support how they feel about it.'

export function systemPrompt(persona, state, memories, extra) {
  const feeling = state.emotions.filter(([k, p]) => p >= 0.15 && k !== 'neutral').map(([k]) => k).join(', ') || 'calm or neutral'
  const parts = [BASE_PROMPT, persona.prompt,
    `What the person seems to be feeling right now (read on-device from their words, and voice and face if they enabled them): ${feeling}; overall mood is ${state.quadrant}. Let this shape your tone. Do not mention scores or that you are reading their emotions unless they ask.`]
  const admitted = memories.filter((m) => m.admitted)
  if (admitted.length) {
    parts.push(`Things the person told you in earlier conversations that relate to this moment:\n${admitted.map((m) => `- ${m.text}`).join('\n')}\nBring one up only if it genuinely helps, and never list them.`)
  }
  return [...parts, ...extra].join('\n\n')
}

// The reflective responder is what Aura says when no language model is loaded. It keys
// off the dominant emotion (more reliable than the arousal estimate) and the stance.
const MOOD_OF = {
  anxious: ['fear', 'nervousness', 'confusion'],
  low: ['sadness', 'grief', 'disappointment', 'remorse', 'embarrassment'],
  angry: ['anger', 'annoyance', 'disapproval', 'disgust'],
  bright: ['joy', 'amusement', 'excitement', 'gratitude', 'love', 'optimism', 'pride', 'relief', 'admiration', 'approval'],
}
const OPENERS = {
  anxious: ['That sounds like a lot of pressure to be carrying.', 'It makes sense to feel on edge with that hanging over you.', 'I can hear how much this is weighing on you.'],
  low: ['That sounds really heavy.', "I'm sorry it feels this way right now.", 'That sounds hard, and it makes sense that it hurts.'],
  angry: ['That sounds genuinely frustrating.', 'I can hear how much that got to you.', 'It makes sense to feel angry about that.'],
  bright: ['I love hearing that.', 'That sounds genuinely good.', "That's lovely. I'm glad you told me."],
  settled: ["I'm here and listening.", 'Thank you for telling me.', "I'm glad you're talking about it."],
}
const FOLLOW = {
  empathetic: ['What part of it is sitting with you the most?', 'How has it been feeling, day to day?'],
  zen: ['Before anything else, try one slow breath with me, letting the exhale run longer than the inhale.', 'Can you notice your feet on the floor for a moment, and let your shoulders drop a little?'],
  strategist: ['What is the thought that keeps coming back, word for word?', 'If a friend said that about themselves, what would you point out to them?'],
  motivator: ['What is one tiny thing you could do in the next ten minutes?', 'What would a small, doable first step look like today?'],
  caregiver: ['Have you had a chance to eat, drink some water or rest today?', 'What would be one kind thing you could do for yourself tonight?'],
  stoic: ["Which part of this is in your hands, and which part isn't?", 'What still matters to you here, even with all of this going on?'],
}
const pick = (a) => a[Math.floor(Math.random() * a.length)]

let lastReflection = ''

export function reflect(state, personaId) {
  let mood = Object.keys(MOOD_OF).find((k) => MOOD_OF[k].includes(state.dominant)) || 'settled'
  // A cheerful label on an unpleasant turn ("I keep thinking I'm going to fail" reads as
  // optimism to GoEmotions) must not get a cheerful reply: trust the valence head instead.
  const pleasant = state.quadrant === 'bright' || state.quadrant === 'calm'
  if (mood === 'bright' && !pleasant) mood = state.quadrant === 'tense' ? 'anxious' : state.quadrant === 'low' ? 'low' : 'settled'
  const trap = 'It sounds like your mind might be jumping to the hardest version of this.'
  const useTrap = state.distortion_type && personaId === 'strategist' && mood !== 'bright' && !lastReflection.startsWith(trap)
  let reply = lastReflection
  for (let i = 0; i < 8 && reply === lastReflection; i++) {
    const opener = useTrap ? trap : pick(OPENERS[mood])
    const follow = mood === 'bright' ? 'What made it feel that way?' : pick(FOLLOW[personaId] || FOLLOW.empathetic)
    reply = `${opener} ${follow}`
  }
  lastReflection = reply
  return reply
}
