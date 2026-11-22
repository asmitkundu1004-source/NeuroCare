import { useEffect, useState } from 'react'
import { Eyebrow, Icon, Reveal, Rise } from '../components/ui.jsx'
import { useEngine } from '../engine/EngineProvider.jsx'

const API = import.meta.env.VITE_API_URL || ''
const own = (url) => {
  try {
    const u = new URL(url, location.href)
    return u.origin === location.origin || (API && u.origin === new URL(API).origin) || ['localhost', '127.0.0.1'].includes(u.hostname)
  } catch {
    return true
  }
}

function useOutbound() {
  const [entries, setEntries] = useState(() => performance.getEntriesByType('resource').filter((e) => !own(e.name)))
  useEffect(() => {
    const po = new PerformanceObserver((list) => {
      const fresh = list.getEntries().filter((e) => !own(e.name))
      if (fresh.length) setEntries((x) => [...x, ...fresh])
    })
    po.observe({ type: 'resource' })
    return () => po.disconnect()
  }, [])
  return entries
}

const STORED = [
  ['Memories', 'Sentences of four words or more that you shared, with their embedding and emotional tone. Used by AGEM to recall context.'],
  ['Mood log', 'For each message: the time, valence, arousal, dominant emotion, safety tier and stance used. Powers Insights.'],
  ['Conversation text', 'Kept so Aura can follow the last few turns of a session.'],
  ['Prosody baseline', 'Three running averages of your speaking energy, pitch movement and rate. No audio.'],
]
const NEVER = ['Camera frames or photos', 'Audio recordings', 'Your name, email or any account', 'Analytics, cookies or trackers']

export default function Privacy() {
  const { engine, status, phase } = useEngine()
  const [stats, setStats] = useState(null)
  const [confirm, setConfirm] = useState(false)
  const [msg, setMsg] = useState(null)
  const outbound = useOutbound()

  useEffect(() => {
    if (engine) engine.stats().then(setStats).catch(() => {})
  }, [engine])

  useEffect(() => {
    if (!confirm) return undefined
    const t = setTimeout(() => setConfirm(false), 5000)
    return () => clearTimeout(t)
  }, [confirm])

  const download = async () => {
    const data = await engine.exportAll()
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }))
    const a = Object.assign(document.createElement('a'), { href: url, download: `neurocare-export-${new Date().toISOString().slice(0, 10)}.json` })
    document.body.appendChild(a)
    a.click()
    a.remove()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const wipe = async () => {
    if (!confirm) {
      setConfirm(true)
      return
    }
    await engine.wipe()
    setConfirm(false)
    setStats(await engine.stats())
    setMsg('Everything has been erased from this device.')
  }

  const hosts = [...new Set(outbound.map((e) => { try { return new URL(e.name).hostname } catch { return e.name } }))]

  return (
    <section className="page">
      <div className="container">
        <div className="page-head">
          <Eyebrow index="Privacy">What NeuroCare keeps, and where</Eyebrow>
          <Rise as="h1" text="Your data has *one* address." immediate />
          <p className="lead">There is no NeuroCare cloud. Depending on how you run it, everything lives either in this browser or in one encrypted file on your own computer.</p>
        </div>
        <div style={{ display: 'grid', gap: 16 }}>
          <div className="grid-3">
            <Reveal className="card stat">
              <span className="v num">{outbound.length}</span>
              <span className="k">requests to other servers since this page opened</span>
              {hosts.length > 0 && <ul className="faint mono" style={{ fontSize: '0.72rem', margin: '8px 0 0', paddingLeft: 16 }}>{hosts.map((h) => <li key={h}>{h}</li>)}</ul>}
            </Reveal>
            <Reveal className="card stat" delay={0.05}>
              <span className="v num">{stats?.memories ?? '—'}</span>
              <span className="k">memories in your vault</span>
            </Reveal>
            <Reveal className="card stat" delay={0.1}>
              <span className="v num">{stats?.turns ?? '—'}</span>
              <span className="k">logged messages, all encrypted</span>
            </Reveal>
          </div>
          <div className="grid-2">
            <Reveal className="card">
              <h3>What is stored</h3>
              <p className="sub">{status?.mode === 'server' ? 'In backend/data/neurocare.db on this computer.' : 'In this browser’s IndexedDB, for this site only.'} Encrypted with {status?.encryption || 'AES-256-GCM'}.</p>
              <div style={{ display: 'grid', gap: 16 }}>
                {STORED.map(([k, v]) => <div key={k}><p className="mono faint" style={{ fontSize: '0.74rem', marginBottom: 4 }}>{k}</p><p>{v}</p></div>)}
              </div>
            </Reveal>
            <Reveal className="card" delay={0.05}>
              <h3>What is never stored</h3>
              <p className="sub">These never leave the page that produced them.</p>
              <ul style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 10 }}>{NEVER.map((n) => <li key={n}>{n}</li>)}</ul>
              <h3 style={{ marginTop: 28 }}>Two honest caveats</h3>
              <ul className="muted" style={{ margin: '10px 0 0', paddingLeft: 18, display: 'grid', gap: 10, fontSize: '0.94rem' }}>
                <li>Voice typing uses your browser’s speech recogniser. Chrome runs it on-device where it can; otherwise it may use the browser vendor’s service. Typing is always fully local.</li>
                <li>The optional in-tab language model downloads its weights once from Hugging Face. No conversation data is sent.</li>
              </ul>
            </Reveal>
          </div>
          <Reveal className="card">
            <h3>Your controls</h3>
            <p className="sub">Take a full copy with you, or erase everything. Erasing cannot be undone.</p>
            <div className="row">
              <button className="btn btn-ghost" disabled={phase !== 'ready'} onClick={download}><Icon name="download" size={16} /> Export everything (JSON)</button>
              <button className="btn btn-danger" disabled={phase !== 'ready'} onClick={wipe}><Icon name="trash" size={16} /> {confirm ? 'Tap again to erase everything' : 'Erase all data'}</button>
            </div>
            {msg && <p className="faint" style={{ marginTop: 14 }} role="status">{msg}</p>}
          </Reveal>
        </div>
      </div>
    </section>
  )
}
