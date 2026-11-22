import { AnimatePresence, motion, useInView, useMotionValue, useScroll, useSpring, useTransform } from 'framer-motion'
import { useEffect, useRef, useState } from 'react'

export const EASE = [0.22, 1, 0.36, 1]

export function Reveal({ children, delay = 0, y = 28, as = 'div', className, ...rest }) {
  const M = motion[as]
  return (
    <M
      className={className}
      initial={{ opacity: 0, y }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '0px 0px -12% 0px' }}
      transition={{ duration: 1.1, ease: EASE, delay }}
      {...rest}
    >
      {children}
    </M>
  )
}

// Headline whose words rise into place one after another. *word* renders in italic serif.
export function Rise({ text, as = 'h2', className = '', delay = 0, stagger = 0.06, immediate = false }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '0px 0px -10% 0px' })
  const show = immediate || inView
  const Tag = as
  const parts = text.split(/(\*[^*]+\*)/g).filter(Boolean)
  let i = 0
  return (
    <Tag ref={ref} className={`rise ${className}`} aria-label={text.replace(/\*/g, '')}>
      {parts.flatMap((part, pi) => {
        const em = part.startsWith('*')
        return part.replace(/\*/g, '').split(/(\s+)/).map((w, wi) => {
          if (!w) return null
          if (/^\s+$/.test(w)) return ' '
          const d = delay + stagger * i++
          return (
            <span className="rise-mask" key={`${pi}-${wi}`} aria-hidden="true">
              <motion.span
                className={em ? 'rise-word em' : 'rise-word'}
                initial={{ y: '112%' }}
                animate={show ? { y: '0%' } : undefined}
                transition={{ duration: 1.05, ease: EASE, delay: d }}
              >
                {w}
              </motion.span>
            </span>
          )
        })
      })}
    </Tag>
  )
}

// Paragraph that brightens word by word as it scrolls through the viewport.
export function ScrollWords({ text, className = '' }) {
  const ref = useRef(null)
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start 85%', 'end 45%'] })
  const words = text.split(' ')
  return (
    <p ref={ref} className={`scroll-words ${className}`}>
      {words.map((w, i) => (
        <Word key={i} progress={scrollYProgress} range={[i / words.length, (i + 1) / words.length]}>{w}</Word>
      ))}
    </p>
  )
}

function Word({ children, progress, range }) {
  const opacity = useTransform(progress, range, [0.16, 1])
  return <motion.span style={{ opacity }}>{children} </motion.span>
}

export function Magnetic({ children, strength = 0.28, className = '' }) {
  const ref = useRef(null)
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const sx = useSpring(x, { stiffness: 220, damping: 18 })
  const sy = useSpring(y, { stiffness: 220, damping: 18 })
  const onMove = (e) => {
    if (e.pointerType !== 'mouse') return
    const r = ref.current.getBoundingClientRect()
    x.set((e.clientX - r.left - r.width / 2) * strength)
    y.set((e.clientY - r.top - r.height / 2) * strength)
  }
  const reset = () => { x.set(0); y.set(0) }
  return (
    <motion.span ref={ref} className={`magnetic ${className}`} style={{ x: sx, y: sy }} onPointerMove={onMove} onPointerLeave={reset}>
      {children}
    </motion.span>
  )
}

export function CountUp({ value, decimals = 0, suffix = '', prefix = '', duration = 1.6 }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  const [shown, setShown] = useState(0)
  useEffect(() => {
    if (!inView || value == null) return undefined
    let raf
    const t0 = performance.now()
    const tick = (now) => {
      const k = Math.min(1, (now - t0) / (duration * 1000))
      setShown(value * (1 - (1 - k) ** 4))
      if (k < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [inView, value, duration])
  return <span ref={ref} className="num">{prefix}{value == null ? '—' : shown.toFixed(decimals)}{suffix}</span>
}

// Russell's circumplex: valence on x, arousal on y, with a trail of recent states.
export function Circumplex({ point, trail = [], size = 220, labels = true, anchors }) {
  const s = size
  const c = s / 2
  const r = s / 2 - 18
  const px = (v) => c + v * r
  const py = (a) => c - a * r
  return (
    <svg className="circumplex" viewBox={`0 0 ${s} ${s}`} width="100%" role="img"
      aria-label={point ? `Valence ${point.valence.toFixed(2)}, arousal ${point.arousal.toFixed(2)}` : 'No reading yet'}>
      <circle cx={c} cy={c} r={r} className="cx-ring" />
      <circle cx={c} cy={c} r={r / 2} className="cx-ring faint" />
      <line x1={c - r} y1={c} x2={c + r} y2={c} className="cx-axis" />
      <line x1={c} y1={c - r} x2={c} y2={c + r} className="cx-axis" />
      {labels && (
        <g className="cx-labels">
          <text x={c} y={11} textAnchor="middle">activated</text>
          <text x={c} y={s - 3} textAnchor="middle">deactivated</text>
          <text x={s - 2} y={c - 6} textAnchor="end">pleasant</text>
          <text x={2} y={c - 6}>unpleasant</text>
        </g>
      )}
      {anchors && Object.entries(anchors).map(([k, [v, a]]) => (
        <g key={k} className="cx-anchor">
          <circle cx={px(v)} cy={py(a)} r={2.2} />
          <text x={px(v) + 4} y={py(a) + 3}>{k}</text>
        </g>
      ))}
      {trail.length > 1 && <polyline className="cx-trail" points={trail.map((t) => `${px(t.valence)},${py(t.arousal)}`).join(' ')} />}
      {point && (
        <motion.g initial={false} animate={{ x: px(point.valence) - c, y: py(point.arousal) - c }} transition={{ type: 'spring', stiffness: 90, damping: 16 }}>
          <circle cx={c} cy={c} r={11} className="cx-glow" />
          <circle cx={c} cy={c} r={4.5} className="cx-dot" />
        </motion.g>
      )}
    </svg>
  )
}

export function Bars({ items, max = 1, format = (v) => v.toFixed(2), highlight = [] }) {
  return (
    <ul className="bars">
      {items.map(([label, v]) => (
        <li key={label} className={highlight.includes(label) ? 'on' : ''}>
          <span className="bars-label">{label}</span>
          <span className="bars-track">
            <motion.span className="bars-fill" initial={false} animate={{ width: `${Math.max(0, Math.min(1, v / max)) * 100}%` }} transition={{ duration: 0.6, ease: EASE }} />
          </span>
          <span className="bars-value">{format(v)}</span>
        </li>
      ))}
    </ul>
  )
}

export function Sheet({ open, onClose, title, children, side = 'right' }) {
  const panel = useRef(null)
  useEffect(() => {
    if (!open) return undefined
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    const prev = document.activeElement
    const t = setTimeout(() => panel.current?.focus(), 50)
    document.documentElement.classList.add('locked')
    return () => {
      clearTimeout(t)
      window.removeEventListener('keydown', onKey)
      document.documentElement.classList.remove('locked')
      prev?.focus?.()
    }
  }, [open, onClose])
  const off = side === 'right' ? { x: '104%' } : { y: '104%' }
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="sheet-root" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.3 }}>
          <div className="sheet-scrim" onClick={onClose} />
          <motion.aside
            ref={panel} tabIndex={-1} role="dialog" aria-modal="true" aria-label={title}
            className={`sheet sheet-${side}`} data-lenis-prevent
            initial={off} animate={{ x: 0, y: 0 }} exit={off}
            transition={{ duration: 0.55, ease: EASE }}
          >
            <header className="sheet-head">
              <h3>{title}</h3>
              <button className="icon-btn" onClick={onClose} aria-label="Close"><Icon name="close" /></button>
            </header>
            <div className="sheet-body">{children}</div>
          </motion.aside>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

const PATHS = {
  close: 'M6 6l12 12M18 6L6 18',
  arrow: 'M5 12h14M13 6l6 6-6 6',
  mic: 'M12 4a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V7a3 3 0 0 1 3-3zM6 11a6 6 0 0 0 12 0M12 17v3',
  send: 'M5 12h13M12 6l6 6-6 6',
  stop: 'M8 8h8v8H8z',
  camera: 'M4 8h3l2-2h6l2 2h3v11H4zM12 17a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z',
  speaker: 'M5 10h3l4-4v12l-4-4H5zM16 9a4 4 0 0 1 0 6M18.5 6.5a7.5 7.5 0 0 1 0 11',
  mute: 'M5 10h3l4-4v12l-4-4H5zM16 10l4 4M20 10l-4 4',
  sliders: 'M5 7h9M18 7h1M5 17h3M12 17h7M14 5v4M8 15v4',
  lock: 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12v9H6z',
  heart: 'M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z',
  eye: 'M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  trash: 'M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12',
  download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
  phone: 'M6 3h4l1.5 5-2.5 1.5a11 11 0 0 0 5.5 5.5L16 12.5l5 1.5v4a2 2 0 0 1-2 2A16 16 0 0 1 4 5a2 2 0 0 1 2-2z',
  info: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5M12 8h.01',
}

export function Icon({ name, size = 18, stroke = 1.6 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  )
}

export function Eyebrow({ children, index }) {
  return (
    <p className="eyebrow">
      {index && <span className="eyebrow-index">{index}</span>}
      {children}
    </p>
  )
}
