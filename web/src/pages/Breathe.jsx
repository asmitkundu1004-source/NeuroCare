import { AnimatePresence, motion } from 'framer-motion'
import { useCallback, useEffect, useRef, useState } from 'react'
import AuraField from '../components/AuraField.jsx'
import { EASE, Eyebrow, Icon } from '../components/ui.jsx'

// each phase: [label, seconds, how full the orb is at the end of the phase]
const PATTERNS = {
  sigh: { name: 'Sigh', note: 'Two inhales, one long exhale. Quick relief when anxious.', phases: [['Breathe in', 2, 0.88], ['A little more', 1, 1], ['Long breath out', 6, 0.55]] },
  box: { name: 'Box', note: 'Four equal sides. Steadies attention under pressure.', phases: [['Breathe in', 4, 1], ['Hold', 4, 1], ['Breathe out', 4, 0.55], ['Hold', 4, 0.55]] },
  478: { name: '4 · 7 · 8', note: 'A long exhale that helps the body settle before sleep.', phases: [['Breathe in', 4, 1], ['Hold', 7, 1], ['Breathe out', 8, 0.55]] },
  coherent: { name: 'Coherent', note: 'About five and a half breaths a minute, slow and even.', phases: [['Breathe in', 5.5, 1], ['Breathe out', 5.5, 0.55]] },
}
const LENGTHS = [1, 3, 5]

function tone(ctx, up) {
  const o = ctx.createOscillator()
  const g = ctx.createGain()
  o.type = 'sine'
  o.frequency.value = up ? 396 : 264
  g.gain.setValueAtTime(0, ctx.currentTime)
  g.gain.linearRampToValueAtTime(0.05, ctx.currentTime + 0.25)
  g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 1.6)
  o.connect(g).connect(ctx.destination)
  o.start()
  o.stop(ctx.currentTime + 1.7)
}

export default function Breathe() {
  const [pattern, setPattern] = useState('sigh')
  const [minutes, setMinutes] = useState(3)
  const [running, setRunning] = useState(false)
  const [phase, setPhase] = useState(0)
  const [left, setLeft] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [sound, setSound] = useState(false)
  const [done, setDone] = useState(false)
  const audio = useRef(null)
  const elapsedRef = useRef(0)

  const phases = PATTERNS[pattern].phases
  const [label, secs, scale] = phases[phase]

  const stop = useCallback(() => {
    setRunning(false)
    setPhase(0)
    setLeft(0)
  }, [])

  useEffect(() => {
    if (!running) return undefined
    const [, s, sc] = phases[phase]
    setLeft(Math.ceil(s))
    if (navigator.vibrate) navigator.vibrate(phase === 0 ? [24, 60, 24] : 18)
    if (sound && audio.current) tone(audio.current, sc >= 1)
    const started = performance.now()
    const tick = setInterval(() => setLeft(Math.max(0, Math.ceil(s - (performance.now() - started) / 1000))), 200)
    const next = setTimeout(() => {
      elapsedRef.current += s
      setElapsed(elapsedRef.current)
      if (phase === phases.length - 1 && elapsedRef.current >= minutes * 60) {
        setRunning(false)
        setDone(true)
        setPhase(0)
      } else {
        setPhase((phase + 1) % phases.length)
      }
    }, s * 1000)
    return () => {
      clearInterval(tick)
      clearTimeout(next)
    }
  }, [running, phase, phases, minutes, sound])

  const start = () => {
    if (sound && !audio.current) audio.current = new AudioContext()
    elapsedRef.current = 0
    setElapsed(0)
    setDone(false)
    setPhase(0)
    setRunning(true)
  }

  const toggleSound = () => {
    if (!sound && !audio.current) audio.current = new AudioContext()
    setSound((s) => !s)
  }

  const remaining = Math.max(0, minutes * 60 - elapsed)
  const fill = running ? scale : 0.72
  const holding = label.startsWith('Hold')

  return (
    <section className="breathe">
      <AuraField affect={{ valence: 0.35, arousal: -0.45 }} level={running ? (scale - 0.55) * 1.4 : 0.1} radius={0.34} center={[0.5, 0.52]} intensity={0.55} />
      <div style={{ position: 'relative', zIndex: 2, display: 'grid', justifyItems: 'center', textAlign: 'center' }}>
        <Eyebrow index="Breathe">{PATTERNS[pattern].name}</Eyebrow>
        <p className="muted" style={{ margin: '12px 0 28px', maxWidth: '28em' }}>{PATTERNS[pattern].note}</p>
        <div className="breath-orb" aria-live="polite">
          <span className="ring" />
          <span className="ring inner" />
          <motion.span className="core" initial={false} animate={{ scale: fill }}
            transition={{ duration: running ? secs : 1.2, ease: holding ? 'linear' : [0.45, 0, 0.55, 1] }} />
          <div className="label">
            <AnimatePresence mode="wait">
              <motion.b key={running ? `${phase}-${label}` : done ? 'done' : 'idle'} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.5, ease: EASE }}>
                {running ? label : done ? 'Well done' : 'Ready'}
              </motion.b>
            </AnimatePresence>
            <span>{running ? left : done ? `${minutes} min complete` : `${minutes} min`}</span>
          </div>
        </div>
        <div className="breathe-controls">
          <div className="segmented" role="radiogroup" aria-label="Pattern">
            {Object.entries(PATTERNS).map(([k, v]) => (
              <button key={k} role="radio" aria-checked={pattern === k} className={pattern === k ? 'on' : ''} onClick={() => { stop(); setPattern(k) }}>{v.name}</button>
            ))}
          </div>
          <div className="row" style={{ justifyContent: 'center' }}>
            <div className="segmented" role="radiogroup" aria-label="Length">
              {LENGTHS.map((m) => <button key={m} role="radio" aria-checked={minutes === m} className={minutes === m ? 'on' : ''} onClick={() => { stop(); setMinutes(m) }}>{m} min</button>)}
            </div>
            <button className={`icon-btn ${sound ? 'on' : ''}`} onClick={toggleSound} aria-pressed={sound} aria-label="Soft tones"><Icon name={sound ? 'speaker' : 'mute'} /></button>
          </div>
          {running ? (
            <button className="btn btn-ghost" onClick={stop}>Stop · {Math.floor(remaining / 60)}:{String(Math.round(remaining % 60)).padStart(2, '0')} left</button>
          ) : (
            <button className="btn btn-primary" onClick={start}>{done ? 'Again' : 'Begin'} <Icon name="arrow" size={16} /></button>
          )}
          <p className="faint" style={{ fontSize: '0.82rem', maxWidth: '30em' }}>If you feel light-headed, stop and breathe normally. On a phone you'll feel a soft tap at each change.</p>
        </div>
      </div>
    </section>
  )
}
