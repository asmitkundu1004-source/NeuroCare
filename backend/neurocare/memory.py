"""Encrypted, single-file episodic memory and mood log.

Everything personal (memory text, memory embeddings, conversation text) is sealed with
AES-256-GCM before it touches disk. Embeddings are encrypted too, because sentence
embeddings can be partially inverted back to text. Decrypted vectors live only in RAM.
At personal scale (thousands of memories) exact search over a numpy matrix is faster
and simpler than an ANN index, and it is exact.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from .agem import AgemParams, rank
from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    created REAL NOT NULL,
    text BLOB NOT NULL,
    embedding BLOB NOT NULL,
    valence REAL NOT NULL,
    arousal REAL NOT NULL,
    dominant TEXT,
    last_turn INTEGER DEFAULT -1000000,
    surfaced INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    session TEXT NOT NULL,
    created REAL NOT NULL,
    role TEXT NOT NULL,
    text BLOB NOT NULL,
    valence REAL,
    arousal REAL,
    dominant TEXT,
    risk_tier INTEGER DEFAULT 0,
    persona TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS turns_created ON turns(created);
"""


class Vault:
    """AES-GCM with a key from a passphrase (scrypt) or a random local key file."""

    def __init__(self, data_dir: Path, passphrase: str | None):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

        if passphrase:
            salt_path = data_dir / "salt.bin"
            if not salt_path.exists():
                salt_path.write_bytes(os.urandom(16))
            key = Scrypt(salt=salt_path.read_bytes(), length=32, n=2 ** 15, r=8, p=1).derive(passphrase.encode())
            self.mode = "passphrase"
        else:
            key_path = data_dir / "vault.key"
            if not key_path.exists():
                key_path.write_bytes(AESGCM.generate_key(bit_length=256))
            key = key_path.read_bytes()
            self.mode = "local key file"
        self._aes = AESGCM(key)

    def seal(self, data: bytes) -> bytes:
        nonce = os.urandom(12)
        return nonce + self._aes.encrypt(nonce, data, None)

    def open(self, blob: bytes) -> bytes:
        return self._aes.decrypt(blob[:12], blob[12:], None)


class MemoryStore:
    def __init__(self, data_dir: Path = settings.data_dir, passphrase: str | None = settings.passphrase,
                 params: AgemParams = AgemParams()):
        self.path = data_dir / "neurocare.db"
        self.vault = Vault(data_dir, passphrase)
        self.params = params
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._load()

    # ------------------------------------------------------------------ cache

    def _load(self):
        with self._lock:
            rows = self.db.execute(
                "SELECT id, created, text, embedding, valence, arousal, dominant, last_turn FROM memories ORDER BY id"
            ).fetchall()
            self.ids = np.array([r[0] for r in rows], np.int64)
            self.created = np.array([r[1] for r in rows], np.float64)
            self.texts = [self.vault.open(r[2]).decode() for r in rows]
            self.emb = (np.vstack([np.frombuffer(self.vault.open(r[3]), np.float32) for r in rows])
                        if rows else np.zeros((0, 384), np.float32))
            self.valence = np.array([r[4] for r in rows], np.float32)
            self.arousal = np.array([r[5] for r in rows], np.float32)
            self.dominant = [r[6] for r in rows]
            self.last_turn = np.array([r[7] for r in rows], np.int64)

    def _meta(self, key: str, default: str = "0") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def next_turn(self) -> int:
        with self._lock:
            n = int(self._meta("turn")) + 1
            self.db.execute("INSERT INTO meta(key, value) VALUES('turn', ?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (str(n),))
            self.db.commit()
            return n

    # ------------------------------------------------------------------ memory

    def remember(self, text: str, embedding: np.ndarray, valence: float, arousal: float, dominant: str):
        now = time.time()
        with self._lock:
            cur = self.db.execute(
                "INSERT INTO memories(created, text, embedding, valence, arousal, dominant) VALUES(?,?,?,?,?,?)",
                (now, self.vault.seal(text.encode()), self.vault.seal(embedding.astype(np.float32).tobytes()),
                 float(valence), float(arousal), dominant))
            self.db.commit()
            self.ids = np.append(self.ids, cur.lastrowid)
            self.created = np.append(self.created, now)
            self.texts.append(text)
            self.emb = np.vstack([self.emb, embedding.reshape(1, -1).astype(np.float32)])
            self.valence = np.append(self.valence, np.float32(valence))
            self.arousal = np.append(self.arousal, np.float32(arousal))
            self.dominant.append(dominant)
            self.last_turn = np.append(self.last_turn, -1_000_000)

    def recall(self, query: np.ndarray, query_valence: float, turn: int) -> list[dict]:
        with self._lock:
            if not len(self.ids):
                return []
            cands = rank(query, self.emb, self.valence, self.arousal,
                         age_days=(time.time() - self.created) / 86400.0,
                         turns_since_surfaced=turn - self.last_turn,
                         query_valence=query_valence, params=self.params)
            out = []
            for c in cands:
                i = c.index
                out.append({
                    "id": int(self.ids[i]), "text": self.texts[i], "similarity": round(c.similarity, 3),
                    "score": round(c.score, 3), "valence": round(float(self.valence[i]), 3),
                    "dominant": self.dominant[i], "created": float(self.created[i]),
                    "admitted": c.gated is None, "gated": c.gated,
                })
                if c.gated is None:
                    self.last_turn[i] = turn
                    self.db.execute("UPDATE memories SET last_turn=?, surfaced=surfaced+1 WHERE id=?",
                                    (turn, int(self.ids[i])))
            self.db.commit()
            return out

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            n = self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,)).rowcount
            self.db.commit()
            self._load()
        return n > 0

    def list_memories(self, limit: int = 200) -> list[dict]:
        with self._lock:
            order = np.argsort(-self.created)[:limit]
            return [{"id": int(self.ids[i]), "text": self.texts[i], "created": float(self.created[i]),
                     "valence": round(float(self.valence[i]), 3), "arousal": round(float(self.arousal[i]), 3),
                     "dominant": self.dominant[i]} for i in order]

    # ------------------------------------------------------------------ mood log

    def log_turn(self, session: str, role: str, text: str, valence: float | None = None, arousal: float | None = None,
                 dominant: str | None = None, risk_tier: int = 0, persona: str | None = None):
        with self._lock:
            self.db.execute(
                "INSERT INTO turns(session, created, role, text, valence, arousal, dominant, risk_tier, persona) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (session, time.time(), role, self.vault.seal(text.encode()), valence, arousal, dominant, risk_tier, persona))
            self.db.commit()

    def history(self, session: str, limit: int) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT role, text FROM turns WHERE session=? ORDER BY id DESC LIMIT ?",
                                   (session, limit)).fetchall()
            return [{"role": r, "content": self.vault.open(t).decode()} for r, t in reversed(rows)]

    def timeline(self, days: int = 30) -> dict:
        since = time.time() - days * 86400
        with self._lock:
            rows = self.db.execute(
                "SELECT created, valence, arousal, dominant, risk_tier, session FROM turns "
                "WHERE role='user' AND created>=? ORDER BY created", (since,)).fetchall()
        points = [{"t": r[0], "valence": r[1], "arousal": r[2], "dominant": r[3], "tier": r[4], "session": r[5]}
                  for r in rows]
        counts: dict[str, int] = {}
        for p in points:
            if p["dominant"]:
                counts[p["dominant"]] = counts.get(p["dominant"], 0) + 1
        return {"points": points, "emotions": counts, "sessions": len({p["session"] for p in points}),
                "memories": int(len(self.ids))}

    # ------------------------------------------------------------------ privacy

    def export_all(self) -> dict:
        with self._lock:
            rows = self.db.execute("SELECT session, created, role, text, valence, arousal, dominant, risk_tier, persona "
                                   "FROM turns ORDER BY id").fetchall()
            turns = [{"session": r[0], "created": r[1], "role": r[2], "text": self.vault.open(r[3]).decode(),
                      "valence": r[4], "arousal": r[5], "dominant": r[6], "risk_tier": r[7], "persona": r[8]}
                     for r in rows]
        return {"exported_at": time.time(), "memories": self.list_memories(limit=10 ** 9), "turns": turns}

    def wipe(self):
        with self._lock:
            self.db.executescript("DELETE FROM memories; DELETE FROM turns; DELETE FROM meta;")
            self.db.commit()
            self.db.execute("VACUUM")
            self._load()

    def stats(self) -> dict:
        with self._lock:
            n_turns = self.db.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        return {"memories": int(len(self.ids)), "turns": int(n_turns),
                "encryption": f"AES-256-GCM ({self.vault.mode})",
                "bytes": self.path.stat().st_size if self.path.exists() else 0}
