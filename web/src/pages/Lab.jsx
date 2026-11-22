import { useEffect, useRef, useState } from 'react'
import { Bars, Circumplex, Eyebrow, Icon, Reveal, Rise } from '../components/ui.jsx'
import { BASE, assess } from '../engine/core.js'
import { useEngine } from '../engine/EngineProvider.jsx'
import { FaceSense } from '../engine/senses.js'

const summaries = import.meta.glob('../data/summary.json', { eager: true, import: 'default' })
const S = Object.values(summaries)[0] || {}

const EXAMPLES = [
  'I finally finished the project and my team loved it',
  "Everyone at work must think I'm useless, I always get things wrong",
  "I can't sleep, my chest feels tight and my mind won't stop racing",
  "I'm so angry they cancelled on me again",
  'Honestly I feel numb, nothing really matters lately',
]
const HIDE_ON_MAP = ['neutral', 'realization', 'approval', 'desire', 'confusion', 'curiosity', 'surprise']

let metaCache = null
const loadMeta = () => (metaCache ||= fetch(`${BASE}models/aura-sense/meta.json`).then((r) => r.json()))

function SensePlayground() {
  const { engine, phase } = useEngine()
  const [text, setText] = useState(EXAMPLES[1])
  const [out, setOut] = useState(null)
  const [meta, setMeta] = useState(null)
  const [anchors, setAnchors] = useState(false)
  const seq = useRef(0)

  useEffect(() => { loadMeta().then(setMeta).catch(() => {}) }, [])
  useEffect(() => {
    if (!engine || !text.trim()) return undefined
    const id = ++seq.current
    const t = setTimeout(async () => {
      try {
        const r = await engine.sense(text)
        if (id === seq.current) setOut(r)
      } catch { /* keep the previous reading */ }
    }, 220)
    return () => clearTimeout(t)
  }, [engine, text])

  const a = out?.affect
  const top = a ? Object.entries(a.emotions).sort((x, y) => y[1] - x[1]).slice(0, 10) : []
  const safety = a && meta ? assess(text, a.risk, meta.risk.threshold_elevated, meta.risk.threshold_crisis) : null
  const shownAnchors = anchors && meta ? Object.fromEntries(Object.entries(meta.circumplex).filter(([k]) => !HIDE_ON_MAP.includes(k))) : null

  return (
    <div className="grid-3">
      <Reveal className="card span-2">
        <h3>Aura-Sense, live</h3>
        <p className="sub">Type anything. One forward pass of the 24 MB model returns everything below, {engine?.kind === 'server' ? 'on the local server' : 'inside this tab'}.</p>
        <textarea className="lab-input" value={text} onChange={(e) => setText(e.target.value)} aria-label="Text to analyse" />
        <div className="row" style={{ margin: '12px 0 22px' }}>
          {EXAMPLES.map((e) => <button key={e} className="starter" onClick={() => setText(e)}>{e.length > 38 ? `${e.slice(0, 36)}…` : e}</button>)}
        </div>
        {phase !== 'ready' ? <div className="empty">Loading Aura-Sense…</div> : !a ? <div className="empty">Waiting for text…</div> : (
          <div className="grid-2">
            <div>
              <div className="field"><span className="label">Top 10 of 28 emotions</span></div>
              <div style={{ marginTop: 12 }}><Bars items={top} highlight={a.active} /></div>
            </div>
            <div style={{ display: 'grid', gap: 18, alignContent: 'start' }}>
              <div>
                <div className="field"><span className="label">Crisis risk · safety tier {safety?.tier ?? 0}</span></div>
                <div className="risk-meter"><i style={{ left: `${(a.risk * 100).toFixed(1)}%` }} /></div>
                <p className="faint mono" style={{ fontSize: '0.76rem' }}>p = {a.risk.toFixed(3)} · {safety?.tier ? safety.reasons[0] : 'no concern'}</p>
              </div>
              <dl className="kv">
                <dt>valence</dt><dd className="mono">{a.valence.toFixed(3)}</dd>
                <dt>arousal</dt><dd className="mono">{a.arousal.toFixed(3)}</dd>
                <dt>thinking</dt><dd>{a.distortion_type ? <span className="tag warm">{a.distortion_type}</span> : <span className="faint">no distortion detected</span>} <span className="faint mono">p {a.distortion.toFixed(2)}</span></dd>
                <dt>active</dt><dd>{a.active.map((x) => <span key={x} className="tag" style={{ marginRight: 4 }}>{x}</span>)}</dd>
                <dt>latency</dt><dd className="mono">{out.ms.toFixed(0)} ms</dd>
              </dl>
            </div>
          </div>
        )}
      </Reveal>
      <Reveal className="card" delay={0.05}>
        <div className="row" style={{ justifyContent: 'space-between' }}>
          <h3>Circumplex</h3>
          <button className="pill" onClick={() => setAnchors((x) => !x)}>{anchors ? 'hide' : 'show'} learned map</button>
        </div>
        <p className="sub">{anchors ? 'Where Aura-Sense placed each emotion, learned only from EmoBank ratings.' : 'Valence and arousal of the text above.'}</p>
        <Circumplex point={a ? { valence: a.valence, arousal: a.arousal } : null} anchors={shownAnchors} size={260} />
      </Reveal>
    </div>
  )
}

function FacePlayground() {
  const video = useRef(null)
  const face = useRef(null)
  const [on, setOn] = useState(false)
  const [r, setR] = useState(null)
  const [err, setErr] = useState(null)

  const stop = () => {
    face.current?.close()
    face.current = null
    video.current?.srcObject?.getTracks().forEach((t) => t.stop())
    if (video.current) video.current.srcObject = null
    setOn(false)
    setR(null)
  }
  useEffect(() => () => {
    face.current?.close()
    video.current?.srcObject?.getTracks().forEach((t) => t.stop())
  }, [])

  const start = async () => {
    setErr(null)
    setOn(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 360 } } })
      video.current.srcObject = stream
      await video.current.play()
      const meta = await loadMeta()
      face.current = await FaceSense.create(video.current, meta.circumplex)
      face.current.start(setR, 12)
    } catch {
      stop()
      setErr('Camera unavailable or permission denied.')
    }
  }

  const probs = r?.present ? Object.entries(r.probs).sort((x, y) => y[1] - x[1]) : []
  const blend = r?.blend ? Object.entries(r.blend).filter(([k]) => k !== '_neutral').sort((x, y) => y[1] - x[1]).slice(0, 8) : []

  return (
    <div className="grid-3">
      <Reveal className="card">
        <h3>Aura-Face, live</h3>
        <p className="sub">The video stays in this tab. MediaPipe turns each frame into 52 numbers, and Aura-Face only ever sees those.</p>
        <div className="cam">
          <video ref={video} playsInline muted style={{ display: on ? 'block' : 'none' }} />
          {!on && <div className="cam-off"><button className="btn btn-ghost btn-sm" onClick={start}><Icon name="camera" size={16} /> Start camera</button></div>}
        </div>
        {on && <button className="btn btn-ghost btn-sm" style={{ marginTop: 14 }} onClick={stop}>Stop camera</button>}
        {err && <p className="faint" style={{ marginTop: 12 }}>{err}</p>}
      </Reveal>
      <Reveal className="card" delay={0.05}>
        <h3>Expression</h3>
        <p className="sub">Ensemble of five small networks, smoothed over frames.</p>
        {probs.length ? <Bars items={probs} highlight={[probs[0][0]]} /> : <div className="empty">{on ? 'Looking for a face…' : 'Camera is off.'}</div>}
      </Reveal>
      <Reveal className="card" delay={0.1}>
        <h3>What the model sees</h3>
        <p className="sub">The strongest of the 52 blendshape coefficients: its entire input.</p>
        {blend.length ? <Bars items={blend} highlight={[blend[0][0]]} /> : <div className="empty">No coefficients yet.</div>}
        {r?.present && <p className="faint mono" style={{ fontSize: '0.76rem', marginTop: 14 }}>valence {r.valence.toFixed(2)} · arousal {r.arousal.toFixed(2)} · confidence {(r.confidence * 100).toFixed(0)}%</p>}
      </Reveal>
    </div>
  )
}

function Table({ t }) {
  return (
    <Reveal className="card">
      <h3>{t.title}</h3>
      <p className="sub">{t.caption}</p>
      <div className="table-wrap">
        <table className="table">
          <thead><tr>{t.columns.map((c, i) => <th key={c} className={i ? 'n' : ''}>{c}</th>)}</tr></thead>
          <tbody>
            {t.rows.map((row, ri) => (
              <tr key={ri} className={t.highlight?.includes(ri) ? 'ours' : ''}>
                {row.map((cell, ci) => <td key={ci} className={ci ? 'n' : ''}>{cell}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Reveal>
  )
}

export default function Lab() {
  return (
    <section className="page">
      <div className="container">
        <div className="page-head">
          <Eyebrow index="Lab">Models trained for NeuroCare</Eyebrow>
          <Rise as="h1" text="Probe the *models* yourself." immediate />
          <p className="lead">Everything here runs live. Below the playgrounds are the evaluation tables, generated by the scripts in <span className="mono">research/</span> from held-out test sets.</p>
        </div>
        <div style={{ display: 'grid', gap: 16 }}>
          <SensePlayground />
          <FacePlayground />
          {(S.tables || []).map((t) => <Table key={t.id} t={t} />)}
        </div>
      </div>
    </section>
  )
}
