import { AnimatePresence, motion } from 'framer-motion'
import Lenis from 'lenis'
import { Component, createContext, lazy, Suspense, useContext, useEffect, useState } from 'react'
import { BrowserRouter, HashRouter, Link, NavLink, Route, Routes, useLocation } from 'react-router-dom'
import { EASE, Icon, Sheet } from './components/ui.jsx'
import { NAV, site } from './content/site.js'
import { REGIONS, resourcesFor } from './engine/core.js'
import { EngineProvider, useEngine } from './engine/EngineProvider.jsx'
import Landing from './pages/Landing.jsx'

// After a redeploy, an open tab still asks for the previous build's file names and the
// import fails. Reload once to pick up the new build instead of hanging on a blank page.
function lazyPage(load) {
  return lazy(() => load().then((m) => { sessionStorage.removeItem('nc.reloaded'); return m }).catch((err) => {
    if (!sessionStorage.getItem('nc.reloaded')) {
      sessionStorage.setItem('nc.reloaded', '1')
      window.location.reload()
      return new Promise(() => {})
    }
    throw err
  }))
}

const Session = lazyPage(() => import('./pages/Session.jsx'))
const Insights = lazyPage(() => import('./pages/Insights.jsx'))
const Breathe = lazyPage(() => import('./pages/Breathe.jsx'))
const Lab = lazyPage(() => import('./pages/Lab.jsx'))
const Privacy = lazyPage(() => import('./pages/Privacy.jsx'))

class PageBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { failed: false }
  }
  static getDerivedStateFromError() { return { failed: true } }
  render() {
    if (!this.state.failed) return this.props.children
    return (
      <section className="page">
        <div className="container page-head">
          <p className="eyebrow">Something went wrong</p>
          <h1>This page didn't load.</h1>
          <p><button className="btn btn-ghost" onClick={() => window.location.reload()}>Reload</button></p>
        </div>
      </section>
    )
  }
}

const HelpCtx = createContext(() => {})
export const useHelp = () => useContext(HelpCtx)

function useLenis(enabled) {
  useEffect(() => {
    if (!enabled || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return undefined
    const lenis = new Lenis({ duration: 1.15, easing: (t) => 1 - Math.pow(1 - t, 4), smoothWheel: true })
    let raf
    const loop = (t) => {
      lenis.raf(t)
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    window.__lenis = lenis
    return () => {
      cancelAnimationFrame(raf)
      lenis.destroy()
      window.__lenis = null
    }
  }, [enabled])
}

function Nav({ onHelp }) {
  const { pathname } = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const [onPaper, setOnPaper] = useState(false)
  useEffect(() => {
    const on = () => {
      setScrolled(window.scrollY > 24)
      const probe = 34
      setOnPaper([...document.querySelectorAll('.paper')].some((el) => {
        const r = el.getBoundingClientRect()
        return r.top <= probe && r.bottom >= probe
      }))
    }
    on()
    window.addEventListener('scroll', on, { passive: true })
    return () => window.removeEventListener('scroll', on)
  }, [])
  useEffect(() => setOpen(false), [pathname])
  useEffect(() => {
    document.documentElement.classList.toggle('locked', open)
    if (open) window.__lenis?.stop()
    else window.__lenis?.start()
    return () => document.documentElement.classList.remove('locked')
  }, [open])

  return (
    <>
      <header className={`nav ${scrolled || open ? 'scrolled' : ''} ${onPaper && !open ? 'on-paper' : ''}`}>
        <div className="container nav-inner">
          <Link to="/" className="wordmark" aria-label="NeuroCare home">NeuroCare <small>Aura {site.version}</small></Link>
          <nav className="nav-links" aria-label="Primary">
            {NAV.map((n) => <NavLink key={n.to} to={n.to} className={({ isActive }) => (isActive ? 'active' : '')}>{n.label}</NavLink>)}
          </nav>
          <div className="nav-cta">
            <button className="pill" onClick={onHelp}><Icon name="phone" size={13} /> Need help now</button>
            <Link to="/session" className="btn btn-primary btn-sm">Begin <Icon name="arrow" size={16} /></Link>
            <button className="icon-btn nav-burger" aria-label={open ? 'Close menu' : 'Open menu'} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
              <Icon name={open ? 'close' : 'sliders'} />
            </button>
          </div>
        </div>
      </header>
      <AnimatePresence>
        {open && (
          <motion.div className="mobile-menu" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.35 }}>
            {[{ to: '/', label: 'Home' }, ...NAV].map((n, i) => (
              <motion.div key={n.to} initial={{ y: 24, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ duration: 0.7, ease: EASE, delay: 0.05 + i * 0.05 }}>
                <Link to={n.to} className="big">{n.label}<small>0{i + 1}</small></Link>
              </motion.div>
            ))}
            <div className="mm-foot">
              <Link to="/session" className="btn btn-primary">Begin a session <Icon name="arrow" size={16} /></Link>
              <button className="btn btn-ghost" onClick={onHelp}><Icon name="phone" size={16} /> Need help now</button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer-grid">
          <div>
            <p className="wordmark" style={{ marginBottom: 14 }}>NeuroCare</p>
            <p style={{ maxWidth: '28em' }}>A privacy-first affective companion. {site.programme}, {site.department}, {site.institute}.</p>
          </div>
          <div>
            <h5>Product</h5>
            <ul>{NAV.slice(0, 3).map((n) => <li key={n.to}><Link to={n.to}>{n.label}</Link></li>)}</ul>
          </div>
          <div>
            <h5>Research</h5>
            <ul>
              <li><Link to="/lab">Model Lab</Link></li>
              <li><Link to="/privacy">Privacy architecture</Link></li>
              <li><a href={site.repo} target="_blank" rel="noreferrer">Source code</a></li>
            </ul>
          </div>
          <div>
            <h5>Team</h5>
            <ul>{site.team.map((t) => <li key={t}>{t}</li>)}<li className="faint">Guided by {site.guide}</li></ul>
          </div>
        </div>
        <div className="footer-note">
          <span>NeuroCare is a research prototype for emotional support. It is not a medical device and does not replace professional care.</span>
          <span>© {site.year} {site.name}</span>
        </div>
      </div>
    </footer>
  )
}

function HelpSheet({ open, onClose }) {
  const { settings, update } = useEngine({ autoload: false })
  return (
    <Sheet open={open} onClose={onClose} title="You don't have to wait">
      <p className="muted">If you are thinking about harming yourself, or you are in danger, please contact someone now. These services are free and confidential.</p>
      <div className="field">
        <span className="label">Region</span>
        <div className="segmented" role="radiogroup" aria-label="Region">
          {REGIONS.map((r) => (
            <button key={r} role="radio" aria-checked={settings.region === r} className={settings.region === r ? 'on' : ''} onClick={() => update({ region: r })}>{r}</button>
          ))}
        </div>
      </div>
      <div>
        {resourcesFor(settings.region).map((r) => (
          <div className="resource" key={r.name}>
            <b>{r.name}</b>
            <span className="c">{r.contact}</span>
          </div>
        ))}
      </div>
      <p className="faint" style={{ fontSize: '0.86rem' }}>If you can, tell someone you trust what is happening and ask them to stay with you.</p>
    </Sheet>
  )
}

function NotFound() {
  return (
    <section className="page">
      <div className="container page-head">
        <p className="eyebrow">404</p>
        <h1>This page drifted off.</h1>
        <p><Link to="/" className="btn btn-ghost">Back home <Icon name="arrow" size={16} /></Link></p>
      </div>
    </section>
  )
}

function Shell() {
  const location = useLocation()
  const immersive = location.pathname.startsWith('/session')
  const [help, setHelp] = useState(false)
  useLenis(!immersive)

  useEffect(() => {
    if (window.__lenis) window.__lenis.scrollTo(0, { immediate: true })
    else window.scrollTo(0, 0)
  }, [location.pathname])

  return (
    <HelpCtx.Provider value={() => setHelp(true)}>
      <div className="grain" aria-hidden="true" />
      {!immersive && <Nav onHelp={() => setHelp(true)} />}
      <PageBoundary key={location.pathname}>
        <motion.main key={location.pathname} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4, ease: EASE }}>
          <Suspense fallback={<div className="page" />}>
            <Routes location={location}>
              <Route path="/" element={<Landing />} />
              <Route path="/session" element={<Session />} />
              <Route path="/insights" element={<Insights />} />
              <Route path="/breathe" element={<Breathe />} />
              <Route path="/lab" element={<Lab />} />
              <Route path="/privacy" element={<Privacy />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
          {!immersive && location.pathname !== '/breathe' && <Footer />}
        </motion.main>
      </PageBoundary>
      <HelpSheet open={help} onClose={() => setHelp(false)} />
    </HelpCtx.Provider>
  )
}

const hash = import.meta.env.VITE_ROUTER === 'hash'
const Router = hash ? HashRouter : BrowserRouter

export default function App() {
  return (
    <EngineProvider>
      <Router basename={hash ? undefined : import.meta.env.BASE_URL.replace(/\/$/, '') || '/'}>
        <Shell />
      </Router>
    </EngineProvider>
  )
}
