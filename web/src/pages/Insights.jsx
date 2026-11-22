import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bars, Circumplex, CountUp, EASE, Eyebrow, Icon, Reveal, Rise } from '../components/ui.jsx'
import { useEngine } from '../engine/EngineProvider.jsx'

const RANGES = [[7, '7 days'], [30, '30 days'], [90, '90 days']]

function Timeline({ points }) {
  const W = 720
  const H = 220
  const P = { l: 34, r: 12, t: 12, b: 26 }
  if (points.length < 2) return <div className="empty">A line appears here after a couple of check-ins.</div>
  const t0 = points[0].t
  const t1 = Math.max(points.at(-1).t, t0 + 1)
  // Over a short span, time on the x axis bunches points together: space them by order instead.
  const byOrder = t1 - t0 < 2 * 86400
  const span = W - P.l - P.r
  const x = (t, i) => P.l + (byOrder ? i / (points.length - 1) : (t - t0) / (t1 - t0)) * span
  const y = (v) => P.t + ((1 - v) / 2) * (H - P.t - P.b)
  // light smoothing so a single noisy turn does not dominate the picture
  const smooth = (key) => points.map((p, i) => {
    const win = points.slice(Math.max(0, i - 2), i + 1).map((q) => q[key] ?? 0)
    return [x(p.t, i), y(win.reduce((a, b) => a + b, 0) / win.length)]
  })
  const path = (pts) => pts.map(([a, b], i) => `${i ? 'L' : 'M'}${a.toFixed(1)},${b.toFixed(1)}`).join(' ')
  const v = smooth('valence')
  const a = smooth('arousal')
  const mid = points[Math.floor((points.length - 1) / 2)].t
  const label = (t) => new Date(t * 1000).toLocaleString(undefined, byOrder ? { hour: 'numeric', minute: '2-digit' } : { day: 'numeric', month: 'short' })
  const dates = (byOrder ? [t0, mid, points.at(-1).t] : [t0, (t0 + t1) / 2, t1]).map(label)
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Mood over time">
      {[1, 0.5, 0, -0.5, -1].map((t) => (
        <g key={t}>
          <line className={t === 0 ? 'zero' : 'gridline'} x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} />
          <text className="tick" x={P.l - 8} y={y(t) + 3} textAnchor="end">{t > 0 ? `+${t}` : t}</text>
        </g>
      ))}
      {dates.map((d, i) => (
        <text key={i} className="tick" x={P.l + (i / 2) * (W - P.l - P.r)} y={H - 6} textAnchor={['start', 'middle', 'end'][i]}>{d}</text>
      ))}
      <motion.path className="series-a" d={path(a)} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, ease: EASE }} />
      <motion.path className="series-v" d={path(v)} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 1.6, ease: EASE }} />
      {v.map(([px, py], i) => (points[i].tier > 0 ? <circle key={i} cx={px} cy={py} r={3.5} fill="var(--danger)" /> : null))}
    </svg>
  )
}

export default function Insights() {
  const { engine, phase, status } = useEngine()
  const [days, setDays] = useState(30)
  const [data, setData] = useState(null)
  const [memories, setMemories] = useState([])

  const refresh = useCallback(async () => {
    if (!engine) return
    const [tl, mem] = await Promise.all([engine.timeline(days), engine.memories()])
    setData(tl)
    setMemories(mem)
  }, [engine, days])

  useEffect(() => { refresh().catch(() => {}) }, [refresh])

  const summary = useMemo(() => {
    if (!data?.points?.length) return null
    const pts = data.points
    const mean = (k) => pts.reduce((s, p) => s + (p[k] ?? 0), 0) / pts.length
    const quads = { bright: 0, calm: 0, tense: 0, low: 0 }
    pts.forEach((p) => {
      const v = p.valence ?? 0
      const a = p.arousal ?? 0
      quads[v >= 0 ? (a >= 0 ? 'bright' : 'calm') : (a >= 0 ? 'tense' : 'low')] += 1
    })
    return { v: mean('valence'), a: mean('arousal'), quads, n: pts.length }
  }, [data])

  const forget = async (id) => {
    await engine.forget(id)
    setMemories((m) => m.filter((x) => x.id !== id))
  }

  const emotions = Object.entries(data?.emotions || {}).sort((a, b) => b[1] - a[1]).slice(0, 8)
  const maxE = emotions[0]?.[1] || 1

  return (
    <section className="page">
      <div className="container">
        <div className="page-head">
          <Eyebrow index="Insights">{status ? `${status.label} · encrypted vault` : 'Opening your vault'}</Eyebrow>
          <Rise as="h1" text="How you've *been*, over time." immediate />
          <div className="row" style={{ justifyContent: 'space-between' }}>
            <p className="lead">Built only from your own check-ins and decrypted on this device for this view. Nothing here is shared or uploaded.</p>
            <div className="switcher" role="tablist" aria-label="Range">
              {RANGES.map(([d, l]) => <button key={d} role="tab" aria-selected={days === d} className={days === d ? 'on' : ''} onClick={() => setDays(d)}>{l}</button>)}
            </div>
          </div>
        </div>

        {phase !== 'ready' ? <div className="empty">Opening your encrypted vault…</div> : !summary ? (
          <Reveal className="card" style={{ textAlign: 'center', padding: '64px 24px' }}>
            <h3>Nothing to show yet</h3>
            <p className="sub" style={{ margin: '8px auto 24px', maxWidth: '30em' }}>Each message you send becomes one private point on this timeline. Come back after a session or two.</p>
            <Link to="/session" className="btn btn-primary">Begin a session <Icon name="arrow" size={16} /></Link>
          </Reveal>
        ) : (
          <div style={{ display: 'grid', gap: 16 }}>
            <div className="grid-4">
              {[[summary.n, 0, 'check-ins'], [data.sessions, 0, 'sessions'], [data.memories, 0, 'memories kept'], [summary.v, 2, 'average valence']].map(([v, d, k]) => (
                <Reveal className="card stat" key={k}><span className="v"><CountUp value={v} decimals={d} /></span><span className="k">{k}</span></Reveal>
              ))}
            </div>
            <div className="grid-3">
              <Reveal className="card span-2">
                <h3>Mood timeline</h3>
                <p className="sub">Smoothed over neighbouring check-ins. Red points mark moments the safety layer raised a concern.</p>
                <Timeline points={data.points} />
                <div className="legend" style={{ marginTop: 14 }}>
                  <span><i style={{ background: 'var(--peach)' }} />valence (pleasant ↔ unpleasant)</span>
                  <span><i style={{ background: 'var(--dusk)' }} />arousal (activated ↔ calm)</span>
                </div>
              </Reveal>
              <Reveal className="card" delay={0.05}>
                <h3>Where you've been</h3>
                <p className="sub">Your recent path and average position on the affect circumplex.</p>
                <Circumplex point={{ valence: summary.v, arousal: summary.a }} trail={data.points.slice(-12).map((p) => ({ valence: p.valence ?? 0, arousal: p.arousal ?? 0 }))} size={230} />
              </Reveal>
            </div>
            <div className="grid-2">
              <Reveal className="card">
                <h3>Most frequent feelings</h3>
                <p className="sub">The dominant emotion Aura-Sense read in each check-in.</p>
                <Bars items={emotions} max={maxE} format={(v) => String(v)} highlight={emotions[0] ? [emotions[0][0]] : []} />
              </Reveal>
              <Reveal className="card" delay={0.05}>
                <h3>Mood quadrants</h3>
                <p className="sub">Share of check-ins in each region of the circumplex.</p>
                <Bars items={Object.entries(summary.quads)} max={summary.n} format={(v) => `${Math.round((v / summary.n) * 100)}%`} />
              </Reveal>
            </div>
            <Reveal className="card">
              <div className="row" style={{ justifyContent: 'space-between', marginBottom: 16 }}>
                <div>
                  <h3>Memory vault</h3>
                  <p className="sub" style={{ margin: 0 }}>What Aura may bring back in future conversations. Forget anything, any time.</p>
                </div>
                <span className="pill"><Icon name="lock" size={12} /> encrypted at rest</span>
              </div>
              {memories.length === 0 ? <div className="empty">No memories yet.</div> : (
                <ul className="vault" data-lenis-prevent>
                  <AnimatePresence initial={false}>
                    {memories.map((m) => (
                      <motion.li key={m.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, x: 24 }} transition={{ duration: 0.35 }}>
                        <div>
                          <p>{m.text}</p>
                          <span>{new Date(m.created * 1000).toLocaleDateString()} · {m.dominant} · valence {Number(m.valence).toFixed(2)}</span>
                        </div>
                        <button className="icon-btn" onClick={() => forget(m.id)} aria-label="Forget this memory" title="Forget"><Icon name="trash" size={16} /></button>
                      </motion.li>
                    ))}
                  </AnimatePresence>
                </ul>
              )}
            </Reveal>
          </div>
        )}
      </div>
    </section>
  )
}
