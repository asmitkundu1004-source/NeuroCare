import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useHelp } from '../App.jsx'
import AuraField from '../components/AuraField.jsx'
import { Bars, Circumplex, EASE, Icon, Sheet } from '../components/ui.jsx'
import { BASE, PERSONA, PERSONAS } from '../engine/core.js'
import { DEVICE_MODELS } from '../engine/engines.js'
import { useEngine } from '../engine/EngineProvider.jsx'
import { Ear, FaceSense, Speaker } from '../engine/senses.js'

const STARTERS = [
  "I can't switch off after work lately",
  "I'm anxious about my exam on Friday",
  'Today was actually a really good day',
  "I keep thinking I'll mess everything up",
]

let circumplexCache = null
const loadCircumplex = () => (circumplexCache ||= fetch(`${BASE}models/aura-sense/meta.json`).then((r) => r.json()).then((m) => m.circumplex).catch(() => null))

function StatusPill({ status, onClick }) {
  if (!status) return null
  return (
    <button className="pill status-pill" onClick={onClick} title="Engine settings">
      <span className={`dot ${status.llmReady ? 'live' : 'warm'}`} />
      <span className="status-text">{status.label} · {status.llm}</span>
    </button>
  )
}

function WhyPanel({ reading, trail }) {
  if (!reading) {
    return <p className="faint" style={{ fontSize: '0.9rem' }}>Say something and this panel will show what Aura sensed, which stance it chose and why, and which memories it brought in or held back.</p>
  }
  const { state, persona, safety, memories, timing, done } = reading
  const ch = state.channels || {}
  return (
    <>
      <div className="panel-block">
        <h4><span>Affect</span><span>{state.quadrant}</span></h4>
        <Circumplex point={state} trail={trail} size={240} />
      </div>
      <div className="panel-block">
        <h4><span>Strongest signals</span></h4>
        <Bars items={state.emotions} highlight={[state.dominant]} />
      </div>
      <div className="panel-block">
        <h4><span>Channels</span></h4>
        <Bars items={[['words', ch.text?.confidence ?? 0], ['face', ch.face?.confidence ?? 0], ['voice', ch.voice?.confidence ?? 0]]}
          format={(v) => (v ? `${Math.round(v * 100)}%` : 'off')} />
      </div>
      <div className="panel-block">
        <h4><span>Stance</span><span>{persona.auto ? 'auto' : 'manual'}</span></h4>
        <p className="reason"><b>{persona.name}</b>, because {persona.reason}.</p>
      </div>
      <div className="panel-block">
        <h4><span>Safety</span><span>tier {safety.tier}</span></h4>
        <p className="reason">{safety.tier === 0 ? 'No concern detected.' : safety.reasons.join('; ')}{safety.medical ? ' Medical boundary applied.' : ''}</p>
      </div>
      <div className="panel-block">
        <h4><span>Memory · AGEM</span><span>{memories.filter((m) => m.admitted).length} recalled</span></h4>
        {memories.length ? (
          <div className="mini-mem">
            {memories.map((m) => (
              <div key={m.id} className={m.admitted ? 'in' : 'held'}>
                {m.text}
                <span>{m.admitted ? `recalled · similarity ${m.similarity.toFixed(2)}` : `held back · ${m.gated}`}</span>
              </div>
            ))}
          </div>
        ) : <p className="faint" style={{ fontSize: '0.86rem' }}>Nothing related yet.</p>}
      </div>
      <div className="panel-block">
        <h4><span>Timing</span></h4>
        <div className="timings">
          <div><b>{timing.sense_ms.toFixed(0)} ms</b><span>sense</span></div>
          <div><b>{timing.recall_ms.toFixed(0)} ms</b><span>recall</span></div>
          <div><b>{done?.first_token_ms ? `${(done.first_token_ms / 1000).toFixed(2)} s` : '…'}</b><span>first word</span></div>
        </div>
      </div>
    </>
  )
}

function SettingsBody({ ctx, onClose }) {
  const { engine, status, settings, update, reconnect } = ctx
  const [loading, setLoading] = useState(null)
  const [err, setErr] = useState(null)
  const isDevice = engine?.kind === 'device'
  const load = async (id) => {
    setErr(null)
    setLoading('Starting…')
    try {
      await engine.loadLLM(id, (p) => setLoading(p.text || 'Loading…'))
      setLoading(null)
    } catch (e) {
      setErr(e.message)
      setLoading(null)
    }
  }
  return (
    <>
      <div className="field">
        <span className="label">Engine</span>
        <div className="segmented">
          {[['auto', 'Automatic'], ['server', 'Local server'], ['device', 'This device']].map(([k, l]) => (
            <button key={k} className={settings.mode === k ? 'on' : ''} onClick={() => { reconnect(k); onClose() }}>{l}</button>
          ))}
        </div>
        <p className="help">Automatic uses the NeuroCare server on this computer when it is running (fine-tuned Aura model, Kokoro voice) and otherwise runs everything in this tab.</p>
      </div>
      {isDevice && (
        <div className="field">
          <span className="label">Language model in this tab</span>
          {status?.webgpu ? (
            <>
              <div className="segmented">
                {DEVICE_MODELS.map((m) => (
                  <button key={m.id} disabled={Boolean(loading)} className={status.llm.startsWith(m.label) ? 'on' : ''} onClick={() => load(m.id)}>{m.label} · {m.size}</button>
                ))}
              </div>
              {loading && <p className="help mono">{loading}</p>}
              {err && <p className="help" style={{ color: 'var(--danger)' }}>{err}</p>}
              <p className="help">Downloaded once from Hugging Face and cached by your browser, then runs on your GPU through WebGPU. Without it, Aura uses a short reflective responder.</p>
            </>
          ) : <p className="help">This browser has no WebGPU, so Aura uses its reflective responder here. Chrome or Edge on a recent laptop can run a full model in the tab.</p>}
        </div>
      )}
      <div className="field">
        <span className="label">Voice</span>
        <div className="toggle-row">
          <div className="t">Speak replies aloud<span>{status?.voice}</span></div>
          <button className="switch" role="switch" aria-checked={settings.voiceOut} onClick={() => update({ voiceOut: !settings.voiceOut })} aria-label="Speak replies aloud" />
        </div>
      </div>
      <div className="field">
        <span className="label">Status</span>
        <dl className="kv">
          <dt>engine</dt><dd>{status?.label}</dd>
          <dt>language</dt><dd>{status?.llm}</dd>
          <dt>memory</dt><dd>{status?.encryption}</dd>
        </dl>
      </div>
    </>
  )
}

export default function Session() {
  const ctx = useEngine()
  const { engine, phase, progress, error, status, settings, update } = ctx
  const openHelp = useHelp()
  const [messages, setMessages] = useState([])
  const [reading, setReading] = useState(null)
  const [trail, setTrail] = useState([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [listening, setListening] = useState(false)
  const [interim, setInterim] = useState('')
  const [camera, setCamera] = useState(false)
  const [faceReading, setFaceReading] = useState(null)
  const [sheet, setSheet] = useState(null)
  const [toast, setToast] = useState(null)
  const [speaking, setSpeaking] = useState(false)

  const speaker = useRef(null)
  if (!speaker.current) speaker.current = new Speaker()
  const ear = useRef(null)
  const face = useRef(null)
  const video = useRef(null)
  const transcript = useRef(null)
  const input = useRef(null)
  const holdStart = useRef(0)
  const toastTimer = useRef(0)

  const flash = useCallback((msg) => {
    clearTimeout(toastTimer.current)
    setToast(msg)
    toastTimer.current = setTimeout(() => setToast(null), 3400)
  }, [])

  useEffect(() => { speaker.current.enabled = settings.voiceOut }, [settings.voiceOut])
  useEffect(() => speaker.current.on(setSpeaking), [])
  useEffect(() => () => {
    speaker.current.stop()
    ear.current?.abort()
    face.current?.close()
    video.current?.srcObject?.getTracks().forEach((t) => t.stop())
    clearTimeout(toastTimer.current)
  }, [])

  useEffect(() => {
    if (!engine) return undefined
    return engine.on((e) => {
      if (e.type === 'affect') {
        setReading({ ...e, done: null })
        setTrail((t) => [...t.slice(-11), { valence: e.state.valence, arousal: e.state.arousal }])
        setMessages((m) => [...m, { role: 'aura', text: '', persona: e.persona, tier: e.safety.tier, resources: e.safety.resources, streaming: true }])
      } else if (e.type === 'token') {
        setMessages((m) => {
          const last = m[m.length - 1]
          if (!last || last.role !== 'aura' || !last.streaming) return m
          return [...m.slice(0, -1), { ...last, text: last.text + e.text }]
        })
      } else if (e.type === 'sentence') {
        speaker.current.say(e.text, e.audio)
      } else if (e.type === 'done') {
        setBusy(false)
        setReading((r) => (r ? { ...r, done: e.timing } : r))
        setMessages((m) => {
          const last = m[m.length - 1]
          if (!last || last.role !== 'aura') return m
          return [...m.slice(0, -1), { ...last, text: e.text, streaming: false, fallback: e.fallback }]
        })
      } else if (e.type === 'interrupted') {
        setBusy(false)
        setMessages((m) => {
          const last = m[m.length - 1]
          if (!last || last.role !== 'aura' || !last.streaming) return m
          return [...m.slice(0, -1), { ...last, streaming: false, interrupted: true }]
        })
      } else if (e.type === 'error') {
        setBusy(false)
        flash(e.message)
        setMessages((m) => {
          const last = m[m.length - 1]
          return last?.streaming ? [...m.slice(0, -1), { ...last, streaming: false }] : m
        })
      }
    })
  }, [engine, flash])

  useEffect(() => {
    const el = transcript.current
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
  }, [messages, interim])

  const interrupt = useCallback(() => {
    speaker.current.stop()
    if (busy) engine?.interrupt()
  }, [busy, engine])

  const send = useCallback((raw, voice = null) => {
    const t = raw.trim()
    if (!t || !engine) return
    speaker.current.stop()
    if (busy) engine.interrupt()
    const f = face.current?.reading()
    const facePayload = f?.present ? { valence: f.valence, arousal: f.arousal, confidence: f.confidence, label: f.label } : null
    setMessages((m) => [...m, { role: 'user', text: t }])
    setText('')
    if (input.current) input.current.style.height = 'auto'
    setBusy(true)
    engine.send({ text: t, persona: settings.persona, face: facePayload, voice })
  }, [busy, engine, settings.persona])

  // ---------------------------------------------------------------- voice in
  const startListening = useCallback(async () => {
    if (ear.current || !engine) return
    if (!Ear.supported) {
      flash('Voice input needs Chrome, Edge or Safari. You can always type instead.')
      return
    }
    interrupt()
    const e = new Ear()
    ear.current = e
    setInterim('')
    setListening(true)
    try {
      await e.start({ onInterim: setInterim })
    } catch {
      ear.current = null
      setListening(false)
      flash('Microphone permission is needed for voice. You can type instead.')
    }
  }, [engine, interrupt, flash])

  const stopListening = useCallback(async () => {
    const e = ear.current
    if (!e) return
    ear.current = null
    setListening(false)
    const { text: heard, prosody } = await e.stop()
    setInterim('')
    if (heard) send(heard, prosody)
    else flash("I didn't catch that. Try holding a little longer, or type.")
  }, [send, flash])

  const onTalkDown = (ev) => {
    ev.preventDefault()
    if (ear.current) {
      stopListening()
      return
    }
    holdStart.current = performance.now()
    startListening()
  }
  // a hold sends on release; a quick tap keeps listening until tapped again
  const onTalkUp = () => {
    if (ear.current && performance.now() - holdStart.current > 350) stopListening()
  }

  useEffect(() => {
    const down = (e) => {
      if (e.code === 'Escape' && !sheet) interrupt()
      if (e.code !== 'Space' || e.repeat || sheet) return
      const el = document.activeElement
      const typing = el === input.current && input.current.value.length > 0
      if (typing || (el && ['INPUT', 'SELECT', 'BUTTON', 'A'].includes(el.tagName))) return
      e.preventDefault()
      holdStart.current = performance.now()
      startListening()
    }
    const up = (e) => {
      if (e.code === 'Space' && ear.current) {
        e.preventDefault()
        stopListening()
      }
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [startListening, stopListening, interrupt, sheet])

  // ---------------------------------------------------------------- camera
  const stopCamera = () => {
    face.current?.close()
    face.current = null
    video.current?.srcObject?.getTracks().forEach((t) => t.stop())
    if (video.current) video.current.srcObject = null
    setCamera(false)
    setFaceReading(null)
  }
  const toggleCamera = async () => {
    if (camera) return stopCamera()
    setCamera(true)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 480 }, height: { ideal: 360 } } })
      video.current.srcObject = stream
      await video.current.play()
      face.current = await FaceSense.create(video.current, await loadCircumplex())
      face.current.start(setFaceReading)
    } catch {
      stopCamera()
      flash('Camera unavailable. Face sensing is optional and everything else still works.')
    }
    return undefined
  }

  const levelFn = useCallback(() => (ear.current ? ear.current.level : speaker.current.level), [])
  const affect = reading?.state || (faceReading?.present ? faceReading : null) || { valence: 0, arousal: 0 }
  const lastAura = [...messages].reverse().find((m) => m.role === 'aura')
  const persona = settings.persona === 'auto' ? null : PERSONA[settings.persona]
  const showBanner = lastAura?.tier > 0 && lastAura.resources?.length > 0

  return (
    <div className="session">
      <div className="session-field"><AuraField affect={affect} level={levelFn} radius={messages.length ? 0.3 : 0.38} center={[0.5, 0.58]} /></div>
      <div className="session-shade" />

      <header className="session-top">
        <div className="left">
          <Link to="/" className="icon-btn" aria-label="Leave session"><Icon name="close" /></Link>
          <StatusPill status={status} onClick={() => setSheet('settings')} />
        </div>
        <div className="right">
          <button className="pill" onClick={() => setSheet('persona')}>
            <span className="dot" style={{ background: persona ? `hsl(${persona.hue} 65% 72%)` : undefined }} />
            {persona ? persona.name : `Auto${reading ? ` · ${reading.persona.name}` : ''}`}
          </button>
          <button className={`icon-btn ${camera ? 'on' : ''}`} onClick={toggleCamera} aria-pressed={camera} aria-label="Face sensing" title="Face sensing, on-device"><Icon name="camera" /></button>
          <button className={`icon-btn ${settings.voiceOut ? '' : 'on'}`} onClick={() => { if (settings.voiceOut) speaker.current.stop(); update({ voiceOut: !settings.voiceOut }) }} aria-label={settings.voiceOut ? 'Mute voice' : 'Unmute voice'}>
            <Icon name={settings.voiceOut ? 'speaker' : 'mute'} />
          </button>
          <button className="icon-btn why-btn" onClick={() => setSheet('why')} aria-label="Why Aura answered this way"><Icon name="info" /></button>
          <button className="icon-btn" onClick={openHelp} aria-label="Need help now"><Icon name="phone" /></button>
        </div>
      </header>

      <div className="banner-slot">
        <AnimatePresence>
          {showBanner && (
            <motion.div className="safety-banner" initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.5, ease: EASE }} role="status">
              <span>{lastAura.tier === 2 ? 'Please reach out to someone now. You deserve real support.' : 'If things feel heavy, talking to someone can help.'}</span>
              <div className="rows">{lastAura.resources.slice(0, 3).map((r) => <span key={r.name}>{r.name}: <b>{r.contact}</b></span>)}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="face-preview" style={{ display: camera ? 'block' : 'none' }}>
        <video ref={video} playsInline muted />
        <span className="tag">{faceReading?.present ? `${faceReading.label} · on-device` : 'finding face…'}</span>
      </div>

      <div className="session-stage">
        <div className="stage-center">
          {phase !== 'ready' ? (
            <div className="loading-screen">
              <div>
                <h2>{phase === 'error' ? 'Something kept Aura from starting.' : 'Waking Aura'}</h2>
                <p>{phase === 'error' ? error : progress || 'Preparing on-device models'}</p>
                {phase === 'error'
                  ? <button className="btn btn-ghost btn-sm" style={{ marginTop: 22 }} onClick={() => ctx.reconnect('device')}>Run in this tab instead</button>
                  : <div className="loader-line" />}
              </div>
            </div>
          ) : (
            <div className="transcript" ref={transcript} aria-live="polite">
              {messages.length === 0 && !listening ? (
                <motion.div className="empty-state" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1, ease: EASE }}>
                  <h2>What's on your mind?</h2>
                  <p>Type, or hold the microphone and talk. Nothing you say leaves this device.</p>
                  <div className="starters">{STARTERS.map((s) => <button key={s} className="starter" onClick={() => send(s)}>{s}</button>)}</div>
                </motion.div>
              ) : (
                messages.map((m, i) => (
                  <motion.div key={i} className={`msg msg-${m.role}`} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE }}>
                    {m.role === 'user' ? <p>{m.text}</p> : (
                      <>
                        {m.text ? <p>{m.text}{m.streaming && <span className="caret" />}</p> : <div className="thinking" aria-label="Aura is thinking"><i /><i /><i /></div>}
                        {!m.streaming && m.persona && (
                          <div className="meta">
                            <span className="tag">{m.persona.name}</span>
                            {m.fallback && <span className="tag warm">reflective responder</span>}
                            {m.interrupted && <span className="tag">interrupted</span>}
                            {m.tier === 2 && <span className="tag alert">safety response</span>}
                          </div>
                        )}
                      </>
                    )}
                  </motion.div>
                ))
              )}
              {listening && <div className="msg msg-user"><p className="faint">{interim || 'Listening…'}</p></div>}
            </div>
          )}
        </div>
        <aside className="insight-panel" aria-label="Why Aura answered this way">
          <WhyPanel reading={reading} trail={trail} />
        </aside>
      </div>

      <div className="composer">
        <div className="composer-inner glass">
          <button className={`talk ${listening ? 'live' : ''}`} onPointerDown={onTalkDown} onPointerUp={onTalkUp} onPointerCancel={onTalkUp}
            onContextMenu={(e) => e.preventDefault()} aria-label={listening ? 'Stop and send' : 'Hold to talk'} disabled={phase !== 'ready'}>
            <Icon name="mic" />
          </button>
          <textarea ref={input} rows={1} value={text} placeholder={listening ? 'Listening…' : 'Write to Aura'} aria-label="Message"
            onChange={(e) => {
              setText(e.target.value)
              e.target.style.height = 'auto'
              e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                send(text)
              }
            }} />
          {busy || speaking ? (
            <button className="send stop" onClick={interrupt} aria-label="Stop"><Icon name="stop" /></button>
          ) : (
            <button className="send" onClick={() => send(text)} disabled={!text.trim() || phase !== 'ready'} aria-label="Send"><Icon name="send" /></button>
          )}
        </div>
        <div className="composer-hint">
          <span><Icon name="lock" size={11} /> Encrypted on this device</span>
          <span className="desk">Hold <span className="kbd">Space</span> to talk · <span className="kbd">Esc</span> to interrupt</span>
        </div>
      </div>

      <Sheet open={sheet === 'persona'} onClose={() => setSheet(null)} title="How should Aura respond?">
        <div className="persona-pick">
          <button className={settings.persona === 'auto' ? 'on' : ''} onClick={() => { update({ persona: 'auto' }); setSheet(null) }}>
            <span className="sw" style={{ background: 'linear-gradient(135deg, var(--aura-1), var(--aura-2), var(--aura-3))' }} />
            <b>Let Aura choose</b><span>Picks a stance each turn from how you seem to feel, and shows why.</span>
          </button>
          {PERSONAS.map((p) => (
            <button key={p.id} className={settings.persona === p.id ? 'on' : ''} style={{ '--h': p.hue }} onClick={() => { update({ persona: p.id }); setSheet(null) }}>
              <span className="sw" /><b>{p.name}</b><span>{p.framework}. {p.blurb}</span>
            </button>
          ))}
        </div>
      </Sheet>
      <Sheet open={sheet === 'settings'} onClose={() => setSheet(null)} title="Settings">
        <SettingsBody ctx={ctx} onClose={() => setSheet(null)} />
      </Sheet>
      <Sheet open={sheet === 'why'} onClose={() => setSheet(null)} title="Why Aura answered this way">
        <WhyPanel reading={reading} trail={trail} />
      </Sheet>

      <AnimatePresence>
        {toast && <motion.div className="toast" initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }} role="status">{toast}</motion.div>}
      </AnimatePresence>
    </div>
  )
}
