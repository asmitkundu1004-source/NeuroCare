// Copies everything the browser runtime needs into public/ so the site is fully
// self-hosted: no CDN, no third-party requests during a session.
//   - Aura-Sense ONNX model, tokenizer and calibration (from backend/models)
//   - the shared safety lexicon
//   - MediaPipe WASM binaries and the face landmarker model (onnxruntime-web's WASM is bundled by Vite)
//   - research summary used by the landing page and the Lab
import { cpSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const repo = resolve(web, '..')
const pub = join(web, 'public')

function copy(from, to, { optional = false } = {}) {
  if (!existsSync(from)) {
    if (!optional) console.warn(`[sync] missing ${from}`)
    return false
  }
  mkdirSync(dirname(to), { recursive: true })
  cpSync(from, to, { recursive: true })
  return true
}

const models = join(repo, 'backend', 'models')
for (const f of ['model.onnx', 'meta.json', 'tokenizer.json']) {
  copy(join(models, 'aura-sense', f), join(pub, 'models', 'aura-sense', f))
}
copy(join(models, 'safety_lexicon.json'), join(web, 'src', 'data', 'safety_lexicon.json'))
copy(join(repo, 'research', 'results', 'summary.json'), join(web, 'src', 'data', 'summary.json'), { optional: true })
copy(join(repo, 'research', 'face', 'face_landmarker.task'), join(pub, 'models', 'face_landmarker.task'), { optional: true })

for (const f of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm']) {
  copy(join(web, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm', f), join(pub, 'mediapipe', f))
}
console.log('[sync] assets ready')
