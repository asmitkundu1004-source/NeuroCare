import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { createEngine } from './engines.js'

const Ctx = createContext(null)

const DEFAULTS = { mode: 'auto', region: 'IN', voiceOut: true, persona: 'auto' }

function loadSettings() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('nc.settings') || '{}') }
  } catch {
    return DEFAULTS
  }
}

export function EngineProvider({ children }) {
  const [settings, setSettings] = useState(loadSettings)
  const [engine, setEngine] = useState(null)
  const [phase, setPhase] = useState('idle') // idle | loading | ready | error
  const [progress, setProgress] = useState('')
  const [error, setError] = useState(null)
  const [tick, bump] = useState(0)
  const pending = useRef(null)
  const settingsRef = useRef(settings)
  settingsRef.current = settings

  const update = useCallback((patch) => {
    setSettings((s) => {
      const next = { ...s, ...patch }
      localStorage.setItem('nc.settings', JSON.stringify(next))
      return next
    })
  }, [])

  // Load on demand, so the landing page never pays for the model download.
  const ensure = useCallback(() => {
    if (pending.current) return pending.current
    setPhase('loading')
    setError(null)
    const { mode, region } = settingsRef.current
    pending.current = createEngine(mode, (p) => setProgress(typeof p === 'string' ? p : p?.text || ''))
      .then((e) => {
        e.setRegion(region)
        setEngine(e)
        setPhase('ready')
        return e
      })
      .catch((err) => {
        setError(err.message || String(err))
        setPhase('error')
        pending.current = null
        throw err
      })
    return pending.current
  }, [])

  const reconnect = useCallback((mode) => {
    engine?.close()
    pending.current = null
    setEngine(null)
    settingsRef.current = { ...settingsRef.current, mode }
    update({ mode })
    setPhase('idle')
  }, [engine, update])

  useEffect(() => {
    if (!engine) return undefined
    engine.setRegion(settings.region)
    return engine.on((e) => { if (e.type === 'status') bump((n) => n + 1) })
  }, [engine, settings.region])

  const value = useMemo(() => ({
    engine, phase, progress, error, ensure, reconnect, settings, update,
    status: engine ? engine.status() : null, tick,
  }), [engine, phase, progress, error, ensure, reconnect, settings, update, tick])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useEngine({ autoload = true } = {}) {
  const ctx = useContext(Ctx)
  const { phase, ensure } = ctx
  useEffect(() => {
    if (autoload && phase === 'idle') ensure().catch(() => {})
  }, [autoload, phase, ensure])
  return ctx
}
