// Encrypted on-device store for the zero-server mode.
// IndexedDB holds AES-GCM ciphertext only. The key is a non-extractable WebCrypto key:
// the browser can use it, but no script (including this one) can read its bytes.

const DB_NAME = 'neurocare'
const VERSION = 1
const enc = new TextEncoder()
const dec = new TextDecoder()

function req(r) {
  return new Promise((resolve, reject) => {
    r.onsuccess = () => resolve(r.result)
    r.onerror = () => reject(r.error)
  })
}

function openDb() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(DB_NAME, VERSION)
    r.onupgradeneeded = () => {
      const db = r.result
      db.createObjectStore('keys')
      db.createObjectStore('meta')
      db.createObjectStore('memories', { keyPath: 'id', autoIncrement: true })
      const turns = db.createObjectStore('turns', { keyPath: 'id', autoIncrement: true })
      turns.createIndex('created', 'created')
      turns.createIndex('session', 'session')
    }
    r.onsuccess = () => resolve(r.result)
    r.onerror = () => reject(r.error)
  })
}

export class Vault {
  static async open() {
    const db = await openDb()
    let key = await req(db.transaction('keys').objectStore('keys').get('vault'))
    if (!key) {
      key = await crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt'])
      await req(db.transaction('keys', 'readwrite').objectStore('keys').put(key, 'vault'))
    }
    const v = new Vault(db, key)
    await v._load()
    return v
  }

  constructor(db, key) {
    this.db = db
    this.key = key
    this.cache = []
  }

  async seal(bytes) {
    const iv = crypto.getRandomValues(new Uint8Array(12))
    return { iv, data: new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, this.key, bytes)) }
  }

  async open(box) {
    return new Uint8Array(await crypto.subtle.decrypt({ name: 'AES-GCM', iv: box.iv }, this.key, box.data))
  }

  store(name, mode = 'readonly') {
    return this.db.transaction(name, mode).objectStore(name)
  }

  async _load() {
    const rows = await req(this.store('memories').getAll())
    this.cache = await Promise.all(rows.map(async (r) => ({
      id: r.id, created: r.created, valence: r.valence, arousal: r.arousal, dominant: r.dominant, lastTurn: r.lastTurn,
      text: dec.decode(await this.open(r.text)),
      embedding: new Float32Array((await this.open(r.embedding)).buffer),
    })))
  }

  async nextTurn() {
    const n = ((await req(this.store('meta').get('turn'))) || 0) + 1
    await req(this.store('meta', 'readwrite').put(n, 'turn'))
    return n
  }

  async remember(text, embedding, valence, arousal, dominant) {
    const row = {
      created: Date.now(), valence, arousal, dominant, lastTurn: -1e6,
      text: await this.seal(enc.encode(text)),
      embedding: await this.seal(new Uint8Array(Float32Array.from(embedding).buffer)),
    }
    const id = await req(this.store('memories', 'readwrite').add(row))
    this.cache.push({ id, created: row.created, valence, arousal, dominant, lastTurn: -1e6, text, embedding: Float32Array.from(embedding) })
  }

  async markSurfaced(ids, turn) {
    for (const id of ids) {
      const s = this.store('memories', 'readwrite')
      const row = await req(s.get(id))
      if (!row) continue
      row.lastTurn = turn
      await req(this.store('memories', 'readwrite').put(row))
      const c = this.cache.find((m) => m.id === id)
      if (c) c.lastTurn = turn
    }
  }

  memories() {
    return this.cache
  }

  async forget(id) {
    await req(this.store('memories', 'readwrite').delete(id))
    this.cache = this.cache.filter((m) => m.id !== id)
  }

  async logTurn(t) {
    const { text, ...rest } = t
    const sealed = await this.seal(enc.encode(text))
    await req(this.store('turns', 'readwrite').add({ ...rest, created: Date.now(), text: sealed }))
  }

  async _turns() {
    const rows = await req(this.store('turns').getAll())
    return Promise.all(rows.map(async (r) => ({ ...r, text: dec.decode(await this.open(r.text)) })))
  }

  async history(session, limit) {
    const rows = (await this._turns()).filter((r) => r.session === session).slice(-limit)
    return rows.map((r) => ({ role: r.role, content: r.text }))
  }

  async timeline(days = 30) {
    const since = Date.now() - days * 86400000
    const rows = (await req(this.store('turns').getAll())).filter((r) => r.role === 'user' && r.created >= since)
    const emotions = {}
    for (const r of rows) if (r.dominant) emotions[r.dominant] = (emotions[r.dominant] || 0) + 1
    return {
      points: rows.map((r) => ({ t: r.created / 1000, valence: r.valence, arousal: r.arousal, dominant: r.dominant, tier: r.risk_tier, session: r.session })),
      emotions, sessions: new Set(rows.map((r) => r.session)).size, memories: this.cache.length,
    }
  }

  async exportAll() {
    return {
      exported_at: Date.now() / 1000,
      memories: this.cache.map(({ embedding: _e, ...m }) => m),
      turns: (await this._turns()).map(({ id: _id, ...t }) => t),
    }
  }

  async wipe() {
    for (const name of ['memories', 'turns', 'meta']) await req(this.store(name, 'readwrite').clear())
    this.cache = []
  }

  async stats() {
    const turns = await req(this.store('turns').count())
    return { memories: this.cache.length, turns, encryption: 'AES-256-GCM (non-extractable browser key)' }
  }
}
