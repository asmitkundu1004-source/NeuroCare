import { useEffect, useRef } from 'react'

// A soft, living field of light rendered with a small fragment shader. Its palette is
// steered by the fused affect state (valence, arousal) and it swells with voice level.

const VERT = 'attribute vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }'

const FRAG = `
precision highp float;
uniform vec2 u_res;
uniform float u_time;
uniform float u_level;
uniform float u_radius;
uniform vec2 u_center;
uniform vec2 u_mouse;
uniform vec3 u_c1;
uniform vec3 u_c2;
uniform vec3 u_c3;
uniform float u_intensity;

float hash(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}
float fbm(vec2 p){
  float v = 0.0, a = 0.5;
  mat2 r = mat2(0.8, -0.6, 0.6, 0.8);
  for (int i = 0; i < 5; i++) { v += a * noise(p); p = r * p * 2.02; a *= 0.5; }
  return v;
}

void main(){
  vec2 uv = gl_FragCoord.xy / u_res;
  float aspect = u_res.x / u_res.y;
  vec2 p = (uv - u_center) * vec2(aspect, 1.0);
  p -= (u_mouse - 0.5) * 0.04;
  float t = u_time * 0.06;

  vec2 q = vec2(fbm(p * 1.6 + t), fbm(p * 1.6 - t + 4.7));
  vec2 r = vec2(fbm(p * 1.3 + 2.0 * q + vec2(1.7, 9.2) + t * 1.4), fbm(p * 1.3 + 2.0 * q + vec2(8.3, 2.8) - t * 1.1));
  float n = fbm(p * 1.1 + 2.4 * r);

  float breathe = 0.035 * sin(u_time * 0.55) + 0.12 * u_level;
  float rad = u_radius * (1.0 + breathe) + 0.11 * (n - 0.5);
  float d = length(p);
  float body = smoothstep(rad, rad * 0.15, d);
  float halo = exp(-pow(max(d - rad * 0.65, 0.0) / (rad * 0.9), 2.0)) * 0.55;

  vec3 col = mix(u_c1, u_c2, smoothstep(0.25, 0.75, n));
  col = mix(col, u_c3, smoothstep(0.45, 0.95, r.x) * 0.65);
  col += 0.12 * vec3(smoothstep(0.55, 0.9, n)) * u_level;

  float a = clamp(body * 0.9 + halo * 0.5, 0.0, 1.0) * u_intensity;
  float g = (hash(gl_FragCoord.xy + fract(u_time)) - 0.5) * 0.035;
  gl_FragColor = vec4(col * a + g * a, a);
}
`

const PALETTES = {
  bright: [[0.98, 0.79, 0.6], [0.95, 0.55, 0.48], [1.0, 0.9, 0.72]],
  calm: [[0.6, 0.86, 0.78], [0.72, 0.89, 0.83], [0.98, 0.9, 0.72]],
  tense: [[0.93, 0.5, 0.42], [0.55, 0.42, 0.85], [0.98, 0.68, 0.5]],
  low: [[0.43, 0.53, 0.8], [0.58, 0.53, 0.8], [0.74, 0.78, 0.93]],
  neutral: [[0.96, 0.82, 0.7], [0.75, 0.7, 0.92], [0.62, 0.84, 0.8]],
}

export function auraPalette(v = 0, a = 0) {
  const mag = Math.min(1, Math.hypot(v, a) * 1.6)
  const corner = v >= 0 ? (a >= 0 ? 'bright' : 'calm') : (a >= 0 ? 'tense' : 'low')
  return PALETTES.neutral.map((c, i) => c.map((x, j) => x + (PALETTES[corner][i][j] - x) * mag))
}

export function auraCss(v = 0, a = 0) {
  const [c1, c2, c3] = auraPalette(v, a).map((c) => `rgb(${c.map((x) => Math.round(x * 255)).join(' ')})`)
  return { c1, c2, c3 }
}

export default function AuraField({ affect, level = 0, radius = 0.42, center = [0.5, 0.5], intensity = 1, className = '', style }) {
  const canvas = useRef(null)
  const live = useRef({ affect, level, radius, center, intensity })
  live.current = { affect, level, radius, center, intensity }

  useEffect(() => {
    const el = canvas.current
    const gl = el.getContext('webgl', { premultipliedAlpha: true, alpha: true, antialias: false, powerPreference: 'low-power' })
    if (!gl) {
      el.dataset.fallback = 'true'
      return undefined
    }
    const compile = (type, src) => {
      const s = gl.createShader(type)
      gl.shaderSource(s, src)
      gl.compileShader(s)
      return s
    }
    const prog = gl.createProgram()
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT))
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG))
    gl.linkProgram(prog)
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      el.dataset.fallback = 'true'
      return undefined
    }
    gl.useProgram(prog)
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer())
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
    const loc = gl.getAttribLocation(prog, 'p')
    gl.enableVertexAttribArray(loc)
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
    const U = Object.fromEntries(['u_res', 'u_time', 'u_level', 'u_radius', 'u_center', 'u_mouse', 'u_c1', 'u_c2', 'u_c3', 'u_intensity']
      .map((n) => [n, gl.getUniformLocation(prog, n)]))

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const mobile = window.matchMedia('(max-width: 760px)').matches
    const dprCap = mobile ? 1 : 1.5
    let w = 1, h = 1
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, dprCap)
      const rect = el.getBoundingClientRect()
      w = Math.max(1, Math.round(rect.width * dpr))
      h = Math.max(1, Math.round(rect.height * dpr))
      el.width = w
      el.height = h
      gl.viewport(0, 0, w, h)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(el)

    const mouse = [0.5, 0.5], mouseT = [0.5, 0.5]
    const onMove = (e) => {
      mouseT[0] = e.clientX / window.innerWidth
      mouseT[1] = 1 - e.clientY / window.innerHeight
    }
    window.addEventListener('pointermove', onMove, { passive: true })

    let visible = true
    const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting })
    io.observe(el)

    const cur = auraPalette(0, 0)
    let lvl = 0
    let rad = live.current.radius
    let raf = 0
    const start = performance.now()
    const frame = (now) => {
      raf = requestAnimationFrame(frame)
      if (!visible || document.hidden) return
      const L = live.current
      const target = auraPalette(L.affect?.valence ?? 0, L.affect?.arousal ?? 0)
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cur[i][j] += (target[i][j] - cur[i][j]) * 0.025
      const lv = typeof L.level === 'function' ? L.level() : L.level
      lvl += ((lv || 0) - lvl) * 0.18
      // radius is in units of height; on portrait screens size it by the width instead
      const aspect = w / h
      const radTarget = L.radius * (aspect < 1 ? Math.max(0.5, aspect * 1.25) : 1)
      rad += (radTarget - rad) * 0.05
      mouse[0] += (mouseT[0] - mouse[0]) * 0.04
      mouse[1] += (mouseT[1] - mouse[1]) * 0.04
      gl.uniform2f(U.u_res, w, h)
      gl.uniform1f(U.u_time, reduced ? 8 : (now - start) / 1000)
      gl.uniform1f(U.u_level, lvl)
      gl.uniform1f(U.u_radius, rad)
      gl.uniform2f(U.u_center, L.center[0], L.center[1])
      gl.uniform2f(U.u_mouse, mouse[0], mouse[1])
      gl.uniform3fv(U.u_c1, cur[0])
      gl.uniform3fv(U.u_c2, cur[1])
      gl.uniform3fv(U.u_c3, cur[2])
      gl.uniform1f(U.u_intensity, L.intensity)
      gl.clearColor(0, 0, 0, 0)
      gl.clear(gl.COLOR_BUFFER_BIT)
      gl.drawArrays(gl.TRIANGLES, 0, 3)
    }
    raf = requestAnimationFrame(frame)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
      io.disconnect()
      window.removeEventListener('pointermove', onMove)
      gl.getExtension('WEBGL_lose_context')?.loseContext()
    }
  }, [])

  return <canvas ref={canvas} className={`aura-field ${className}`} style={style} aria-hidden="true" />
}
