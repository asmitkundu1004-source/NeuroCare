import { AnimatePresence, motion, useScroll, useTransform } from 'framer-motion'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import AuraField from '../components/AuraField.jsx'
import { Bars, CountUp, EASE, Eyebrow, Icon, Magnetic, Reveal, Rise, ScrollWords } from '../components/ui.jsx'
import { site } from '../content/site.js'
import { PERSONAS } from '../engine/core.js'

const summaries = import.meta.glob('../data/summary.json', { eager: true, import: 'default' })
const S = Object.values(summaries)[0] || {}
const sense = S.aura_sense || {}
const face = S.aura_face || {}
const agem = S.agem || {}

// ---------------------------------------------------------------- hero

const MOODS = [{ valence: 0.25, arousal: 0.1 }, { valence: 0.35, arousal: -0.3 }, { valence: -0.3, arousal: -0.25 }, { valence: 0.4, arousal: 0.35 }]

function Hero() {
  const ref = useRef(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] })
  const y = useTransform(scrollYProgress, [0, 1], ['0%', '18%'])
  const fade = useTransform(scrollYProgress, [0, 0.8], [1, 0])
  const [mood, setMood] = useState(MOODS[0])
  useEffect(() => {
    let i = 0
    const t = setInterval(() => {
      i = (i + 1) % MOODS.length
      setMood(MOODS[i])
    }, 5200)
    return () => clearInterval(t)
  }, [])

  return (
    <section className="hero" ref={ref}>
      <motion.div className="hero-field" style={{ y }}>
        <AuraField affect={mood} radius={0.5} center={[0.66, 0.6]} />
      </motion.div>
      <div className="hero-vignette" />
      <motion.div className="container hero-content" style={{ opacity: fade }}>
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.2, delay: 0.2 }}>
          <Eyebrow index="NC—02">Affective computing, on your own device</Eyebrow>
        </motion.div>
        <Rise as="h1" immediate delay={0.25} stagger={0.07} text="A companion that *listens* to how you feel, and keeps it to itself." />
        <div className="hero-sub">
          <motion.p className="lead" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1.2, ease: EASE, delay: 1.0 }}>
            NeuroCare reads emotion from your words, voice and face with models trained for this project, chooses how to respond,
            remembers what matters and stays quiet about what hurts. Every step runs here, on this device.
          </motion.p>
          <motion.div className="hero-actions" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1.2, ease: EASE, delay: 1.15 }}>
            <Magnetic><Link to="/session" className="btn btn-primary">Begin a session <Icon name="arrow" size={16} /></Link></Magnetic>
            <Magnetic><a href="#loop" className="btn btn-ghost" onClick={(e) => { if (window.__lenis) { e.preventDefault(); window.__lenis.scrollTo('#loop', { offset: -40 }) } }}>How it works</a></Magnetic>
          </motion.div>
        </div>
      </motion.div>
      <motion.div className="container hero-foot" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 1.4, delay: 1.4 }}>
        <div className="hero-facts">
          <div><b>0 bytes</b><span>sent to the cloud</span></div>
          <div><b>3 models</b><span>trained for NeuroCare</span></div>
          <div><b>{sense.size_mb ? `${sense.size_mb} MB` : '24 MB'}</b><span>affect model, runs offline</span></div>
        </div>
        <span className="scroll-cue"><i />Scroll</span>
      </motion.div>
    </section>
  )
}

// ---------------------------------------------------------------- manifesto

function Manifesto() {
  return (
    <section className="section paper">
      <div className="container manifesto">
        <Eyebrow index="01">Why this exists</Eyebrow>
        <ScrollWords text="Most mental-health apps ask you to trust a server with the most private sentences you will ever write. NeuroCare is built the other way round. The listening, the remembering and the deciding all happen on the device in your hands, and the system is designed to know when not to bring something up." />
        <div className="manifesto-foot">
          {[
            ['Private by construction', 'No account, no analytics, no cloud model. Memory is encrypted at rest with a key that never leaves the device.'],
            ['Affect is the control signal', 'What you seem to feel decides the stance Aura takes, what it remembers and when it steps aside for a human.'],
            ['Honest about limits', 'A crisis bypasses the language model entirely and points to real people. Aura never diagnoses or advises on medication.'],
          ].map(([h, p], i) => (
            <Reveal key={h} delay={i * 0.08}>
              <span className="n">0{i + 1}</span>
              <h4>{h}</h4>
              <p>{p}</p>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- the closed loop

const STEPS = [
  { key: 'Sense', title: 'One pass, every signal', body: 'Your words go through Aura-Sense once. That single forward pass returns the memory embedding and the full affect reading: 28 emotions, valence and arousal, crisis risk and thinking patterns.', tech: ['Aura-Sense · MiniLM-L6', 'runs in WebAssembly'] },
  { key: 'Safeguard', title: 'Safety runs first', body: 'A lexicon and the risk head assess every message before any text is generated. In a crisis the language model is bypassed and a reviewed message with helplines is shown instead.', tech: ['3 tiers', 'sticky for 4 turns'] },
  { key: 'Fuse', title: 'Words, voice and face agree', body: 'Optional voice prosody and facial blendshapes are fused with the text reading by confidence, then smoothed across turns, on one shared valence–arousal plane.', tech: ['Aura-Face', 'prosody baseline'] },
  { key: 'Route', title: 'The right stance, explained', body: 'A router picks one of six therapeutic stances (grounding, reflection, reframing and more) from the fused state, and tells you why it chose it.', tech: ['6 stances', 'auto or manual'] },
  { key: 'Remember', title: 'Memory that knows when to stay quiet', body: 'Affect-Gated Episodic Memory ranks past moments by relevance, emotional salience and recency, and holds back painful ones unless they are truly the topic.', tech: ['AGEM', 'AES-256-GCM'] },
  { key: 'Respond', title: 'A voice at conversational pace', body: 'The fine-tuned Aura model streams a reply under the chosen stance. Speech starts after the first sentence, and you can interrupt at any time.', tech: ['Llama 3.2 3B LoRA', 'Kokoro-82M'] },
]

function Loop() {
  const [active, setActive] = useState(0)
  const refs = useRef([])
  useEffect(() => {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => { if (e.isIntersecting) setActive(Number(e.target.dataset.i)) })
    }, { rootMargin: '-45% 0px -45% 0px' })
    refs.current.forEach((el) => el && io.observe(el))
    return () => io.disconnect()
  }, [])
  const R = 200
  const C = 260
  const circ = 2 * Math.PI * R
  const nodes = STEPS.map((s, i) => {
    const a = (i / STEPS.length) * Math.PI * 2 - Math.PI / 2
    return { ...s, a, x: C + R * Math.cos(a), y: C + R * Math.sin(a) }
  })
  return (
    <section className="section" id="loop">
      <div className="container">
        <div className="head">
          <div>
            <Eyebrow index="02">The novelty</Eyebrow>
            <Rise text="A *closed* affective loop." />
          </div>
          <Reveal><p className="lead">Most companions treat emotion as something to display. NeuroCare treats it as a control signal: every turn, what you feel steers safety, stance, memory and voice.</p></Reveal>
        </div>
        <div className="loop">
          <div className="loop-sticky">
            <div className="loop-ring">
              <svg viewBox="0 0 520 520">
                <defs>
                  <linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stopColor="#f5d1b3" />
                    <stop offset="0.55" stopColor="#bfb3ec" />
                    <stop offset="1" stopColor="#9ed6c8" />
                  </linearGradient>
                </defs>
                <circle cx={C} cy={C} r={R} className="ring-track" />
                <motion.circle cx={C} cy={C} r={R} className="ring-progress" transform={`rotate(-90 ${C} ${C})`}
                  strokeDasharray={circ} initial={false} animate={{ strokeDashoffset: circ * (1 - (active + 1) / STEPS.length) }} transition={{ duration: 1, ease: EASE }} />
                {nodes.map((n, i) => {
                  const lx = C + (R + 30) * Math.cos(n.a)
                  const ly = C + (R + 30) * Math.sin(n.a)
                  const anchor = Math.abs(Math.cos(n.a)) < 0.2 ? 'middle' : Math.cos(n.a) > 0 ? 'start' : 'end'
                  return (
                    <g key={n.key} className={`ring-node ${i <= active ? 'on' : ''}`}>
                      <circle cx={n.x} cy={n.y} r={i === active ? 8 : 5} />
                      <text x={lx} y={ly + 4} textAnchor={anchor}>{n.key}</text>
                    </g>
                  )
                })}
              </svg>
              <div className="ring-center">
                <AnimatePresence mode="wait">
                  <motion.div key={active} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.45, ease: EASE }}>
                    <div className="big">{STEPS[active].key}</div>
                    <div className="small">step {active + 1} of {STEPS.length}</div>
                  </motion.div>
                </AnimatePresence>
              </div>
            </div>
          </div>
          <div className="loop-steps">
            {STEPS.map((s, i) => (
              <div key={s.key} data-i={i} ref={(el) => { refs.current[i] = el }} className={`loop-step ${i === active ? 'on' : ''}`}>
                <span className="n">0{i + 1} · {s.key}</span>
                <h3>{s.title}</h3>
                <p>{s.body}</p>
                <div className="tech">{s.tech.map((t) => <span className="pill" key={t}>{t}</span>)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- three senses

function TextViz() {
  const samples = S.samples?.length ? S.samples : null
  const [i, setI] = useState(0)
  useEffect(() => {
    if (!samples) return undefined
    const t = setInterval(() => setI((x) => (x + 1) % samples.length), 3800)
    return () => clearInterval(t)
  }, [samples])
  if (!samples) return null
  const s = samples[i]
  return (
    <div className="viz-text">
      <AnimatePresence mode="wait">
        <motion.p key={i} className="line" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.6, ease: EASE }}>
          “{s.text}”
        </motion.p>
      </AnimatePresence>
      <Bars items={s.top} highlight={[s.top[0][0]]} />
    </div>
  )
}

function FaceViz() {
  const dots = useMemo(() => {
    const out = []
    const ring = (n, rx, ry, cx, cy, start = 0, end = Math.PI * 2) => {
      for (let i = 0; i < n; i++) {
        const a = start + ((end - start) * i) / n
        out.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)])
      }
    }
    ring(20, 44, 54, 50, 50)
    ring(6, 8, 4, 33, 40)
    ring(6, 8, 4, 67, 40)
    ring(5, 10, 2, 33, 31, Math.PI, Math.PI * 2)
    ring(5, 10, 2, 67, 31, Math.PI, Math.PI * 2)
    ring(5, 4, 9, 50, 52, -Math.PI / 2, Math.PI / 2)
    ring(5, 13, 5, 50, 72, 0, Math.PI)
    return out.slice(0, 52)
  }, [])
  return (
    <div className="viz-face" aria-hidden="true">
      {dots.map(([x, y], i) => (
        <motion.span key={i} style={{ left: `${x}%`, top: `${y}%` }}
          animate={{ opacity: [0.25, 1, 0.25], scale: [1, 1.6, 1] }}
          transition={{ duration: 3.2, repeat: Infinity, delay: (i % 13) * 0.18, ease: 'easeInOut' }} />
      ))}
    </div>
  )
}

function VoiceViz() {
  return (
    <div className="viz-wave" aria-hidden="true">
      {Array.from({ length: 28 }, (_, i) => (
        <motion.i key={i} animate={{ height: [10, 26 + ((i * 37) % 90), 14] }}
          transition={{ duration: 1.4 + (i % 5) * 0.2, repeat: Infinity, repeatType: 'mirror', ease: 'easeInOut', delay: i * 0.04 }} />
      ))}
    </div>
  )
}

function Senses() {
  return (
    <section className="section" style={{ paddingTop: 0 }}>
      <div className="container">
        <div className="head">
          <div>
            <Eyebrow index="03">Three senses</Eyebrow>
            <Rise text="Read *carefully*, nothing uploaded." />
          </div>
          <Reveal><p className="lead">Each channel is optional, each runs in the browser, and only a few numbers reach the conversation. Video frames and audio never leave the page.</p></Reveal>
        </div>
        <div className="senses">
          <Reveal className="sense-card">
            <Eyebrow>Words</Eyebrow>
            <div className="viz"><TextViz /></div>
            <h3>Aura-Sense</h3>
            <p>A sentence encoder fine-tuned on four tasks at once while held to its original geometry, so the same vector powers emotion, risk and memory search.</p>
            <div className="spec">
              <span className="pill">28 emotions · F1 {sense.emotion_macro_f1?.toFixed(2) ?? '—'}</span>
              <span className="pill">risk AUROC {sense.risk_auroc?.toFixed(3) ?? '—'}</span>
            </div>
          </Reveal>
          <Reveal className="sense-card" delay={0.08}>
            <Eyebrow>Face</Eyebrow>
            <div className="viz"><FaceViz /></div>
            <h3>Aura-Face</h3>
            <p>Classifies expression from 52 facial muscle coefficients rather than pixels, so the model never sees who you are, only how your face is moving.</p>
            <div className="spec">
              <span className="pill">8 expressions · {face.accuracy ? `${(face.accuracy * 100).toFixed(1)}%` : '—'} accuracy</span>
              <span className="pill">no pixels</span>
            </div>
          </Reveal>
          <Reveal className="sense-card" delay={0.16}>
            <Eyebrow>Voice</Eyebrow>
            <div className="viz"><VoiceViz /></div>
            <h3>Prosody</h3>
            <p>Energy, pitch movement and speaking rate, measured against your own baseline over time, so a naturally quiet voice is not mistaken for a sad one.</p>
            <div className="spec">
              <span className="pill">arousal channel</span>
              <span className="pill">personal baseline</span>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- personas + router

function Personas() {
  const routes = S.routes || []
  const [i, setI] = useState(0)
  useEffect(() => {
    if (!routes.length) return undefined
    const t = setInterval(() => setI((x) => (x + 1) % routes.length), 4200)
    return () => clearInterval(t)
  }, [routes.length])
  const r = routes[i]
  return (
    <section className="section" style={{ paddingTop: 0 }}>
      <div className="container">
        <div className="head">
          <div>
            <Eyebrow index="04">Six stances</Eyebrow>
            <Rise text="Not one voice. The *right* one." />
          </div>
          <Reveal><p className="lead">Each stance comes from an established approach to care. Choose one yourself, or let Aura choose from how you seem to feel, with its reason shown every time.</p></Reveal>
        </div>
        <div className="personas">
          {PERSONAS.map((p, k) => (
            <Reveal key={p.id} className="persona" style={{ '--h': p.hue }} delay={(k % 3) * 0.06}>
              <span className="idx">0{k + 1}</span>
              <h3>{p.name}</h3>
              <span className="fw">{p.framework}</span>
              <p>{p.blurb}</p>
            </Reveal>
          ))}
        </div>
        {r && (
          <Reveal className="router-demo glass">
            <AnimatePresence mode="wait">
              <motion.p key={`q${i}`} className="quote" initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.5, ease: EASE }}>“{r.text}”</motion.p>
            </AnimatePresence>
            <AnimatePresence mode="wait">
              <motion.div key={`d${i}`} className="decision" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.4, delay: 0.15 }}>
                <span>reading → <b>{r.dominant}</b>, mood {r.quadrant}</span>
                <span>stance → <b>{r.persona_name}</b></span>
                <span>because → {r.reason}</span>
              </motion.div>
            </AnimatePresence>
          </Reveal>
        )}
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- memory

function Memory() {
  const [mode, setMode] = useState('good')
  const demo = agem.demo
  if (!demo) return null
  const view = demo[mode]
  return (
    <section className="section" style={{ paddingTop: 0 }}>
      <div className="container">
        <div className="head">
          <div>
            <Eyebrow index="05">Affect-gated memory</Eyebrow>
            <Rise text="Remembers what matters. *Knows* when to let it rest." />
          </div>
          <Reveal><p className="lead">Plain retrieval keeps resurfacing a painful week on a good day. AGEM checks how you feel before deciding which memories enter the conversation.</p></Reveal>
        </div>
        <div className="memory-stage">
          <Reveal>
            <div className="switcher" role="tablist" aria-label="Scenario">
              {[['good', 'On a good day'], ['hard', 'On a hard day']].map(([k, l]) => (
                <button key={k} role="tab" aria-selected={mode === k} className={mode === k ? 'on' : ''} onClick={() => setMode(k)}>{l}</button>
              ))}
            </div>
            <AnimatePresence mode="wait">
              <motion.p key={mode} className="mem-query" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.5, ease: EASE }}>“{view.query}”</motion.p>
            </AnimatePresence>
            <p className="muted">Output of the real AGEM implementation on Aura-Sense embeddings.{agem.isr_plain != null ? ` In the full benchmark, plain retrieval brought up a painful memory on ${Math.round(agem.isr_plain * 100)}% of happy or unrelated messages; AGEM on ${Math.round(agem.isr_agem * 100)}%.` : ''}</p>
          </Reveal>
          <div className="mem-list">
            {view.items.map((m) => (
              <motion.div layout key={m.text} className={`mem ${m.admitted ? 'in' : 'held'}`} transition={{ duration: 0.6, ease: EASE }}>
                <span className="t">{m.text}</span>
                <span className="sim">{m.similarity.toFixed(2)}</span>
                <span className="why">{m.admitted ? 'brought into the conversation' : `held back: ${m.gated}`}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- results

function Results() {
  const items = [
    { v: sense.risk_auroc, d: 3, k: 'Crisis-risk AUROC', note: `Held-out posts, n = ${(sense.risk_n || 10000).toLocaleString()}. A lexicon gate covers short, explicit messages.` },
    { v: sense.valence_r_circ ?? sense.valence_r, d: 2, k: 'Valence correlation (r)', note: 'EmoBank test set, read through the learned circumplex.' },
    { v: sense.circ_nrc_valence_r, d: 2, k: 'Learned map vs human norms', note: 'Emotion positions learned from data agree with NRC-VAD ratings the model never saw.' },
    { v: sense.stsb, d: 3, k: 'Retrieval geometry kept', note: `STS-B Spearman after multi-task training (base model ${sense.stsb_teacher?.toFixed(3) ?? '—'}).` },
  ]
  return (
    <section className="section">
      <div className="container">
        <div className="head">
          <div>
            <Eyebrow index="06">Measured, not claimed</Eyebrow>
            <Rise text="Numbers from *held-out* data." />
          </div>
          <Reveal><p className="lead">Every figure here comes from the training and evaluation scripts in the repository, on test sets the models never trained on. The Model Lab lets you probe them live.</p></Reveal>
        </div>
        <div className="metrics">
          {items.map((m, i) => (
            <Reveal className="metric" key={m.k} delay={i * 0.06}>
              <span className="v"><CountUp value={m.v} decimals={m.d} /></span>
              <span className="k">{m.k}</span>
              <span className="d">{m.note}</span>
            </Reveal>
          ))}
        </div>
        <Reveal style={{ marginTop: 32 }}><Link to="/lab" className="btn btn-ghost">Open the Model Lab <Icon name="arrow" size={16} /></Link></Reveal>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------- privacy boundary

function Boundary() {
  const chips = [
    ['Aura-Sense', 'ONNX · WebAssembly'], ['Aura-Face', 'MediaPipe blendshapes'], ['Safety gate', 'lexicon + risk head'], ['AGEM memory', 'exact cosine search'],
    ['Vault', 'AES-256-GCM, non-extractable key'], ['Language model', 'Ollama or WebGPU'], ['Voice', 'Kokoro or system voice'], ['Mood timeline', 'stays in the vault'],
  ]
  return (
    <section className="section" style={{ paddingTop: 0 }}>
      <div className="container">
        <div className="head">
          <div>
            <Eyebrow index="07">Privacy architecture</Eyebrow>
            <Rise text="Everything lives *inside* the line." />
          </div>
          <Reveal><p className="lead">Run it with the local server for the fine-tuned model and neural voice, or entirely in the browser. Either way the boundary is the same.</p></Reveal>
        </div>
        <Reveal className="device">
          <span className="device-label">Your device</span>
          <div className="device-grid">
            {chips.map(([b, s]) => <div className="chip" key={b}><b>{b}</b><span>{s}</span></div>)}
          </div>
        </Reveal>
        <Reveal className="outside"><span>the internet</span><span className="line" /><span>nothing crosses during a session</span></Reveal>
      </div>
    </section>
  )
}

function Closing() {
  return (
    <section className="closing">
      <AuraField affect={{ valence: 0.3, arousal: -0.2 }} radius={0.36} center={[0.5, 0.45]} />
      <div className="container">
        <Rise text="Say what's *on your mind.*" />
        <div className="actions">
          <Magnetic><Link to="/session" className="btn btn-primary">Begin a session <Icon name="arrow" size={16} /></Link></Magnetic>
          <Magnetic><Link to="/breathe" className="btn btn-ghost">Just breathe for a minute</Link></Magnetic>
        </div>
        <p className="note">{site.programme} · {site.department}, {site.institute}. NeuroCare supports, and does not replace, professional care.</p>
      </div>
    </section>
  )
}

export default function Landing() {
  return (
    <>
      <Hero />
      <Manifesto />
      <Loop />
      <Senses />
      <Personas />
      <Memory />
      <Results />
      <Boundary />
      <Closing />
    </>
  )
}
