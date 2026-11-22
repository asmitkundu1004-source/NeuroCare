"""Aura-Sense runtime: a single ONNX pass yields the memory embedding and the affect reading."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import settings


@dataclass
class TextAffect:
    emotions: dict[str, float]
    active: list[str]
    ekman: dict[str, float]
    valence: float
    arousal: float
    risk: float
    distortion: float
    distortion_type: str | None

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        return sorted(self.emotions.items(), key=lambda kv: -kv[1])[:k]


@dataclass
class Reading:
    embedding: np.ndarray
    affect: TextAffect


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class AuraSense:
    def __init__(self, model_dir: Path = settings.model_dir):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.labels: list[str] = self.meta["emotion"]["labels"]
        self.thresholds = np.array(self.meta["emotion"]["thresholds"], np.float32)
        self.ekman: dict[str, list[str]] = self.meta["emotion"]["ekman"]
        self.circumplex = self.meta["circumplex"]
        self.risk_elevated = self.meta["risk"]["threshold_elevated"]
        self.risk_crisis = self.meta["risk"]["threshold_crisis"]
        self.distortion_threshold = self.meta["distortion"]["threshold"]
        self.distortion_types: list[str] = self.meta["distortion"]["types"]

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=self.meta.get("max_length", 128))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.threads
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()

    def _run(self, texts: list[str]):
        enc = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in enc], np.int64),
            "attention_mask": np.array([e.attention_mask for e in enc], np.int64),
        }
        with self._lock:
            return self.session.run(None, feed)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._run(texts)[0].astype(np.float32) if texts else np.zeros((0, 384), np.float32)

    def read(self, texts: list[str] | str) -> list[Reading]:
        if isinstance(texts, str):
            texts = [texts]
        emb, emo, va, risk, dist, dtype = self._run(texts)
        p_all, r_all, d_all = _sigmoid(emo), _softmax(risk)[:, 1], _softmax(dist)[:, 1]
        out = []
        for i in range(len(texts)):
            p = p_all[i]
            active = [l for l, pi, t in zip(self.labels, p, self.thresholds) if pi >= t]
            ekman = {k: round(float(max(p[self.labels.index(m)] for m in ms)), 4) for k, ms in self.ekman.items()}
            d = float(d_all[i])
            out.append(Reading(
                embedding=emb[i].astype(np.float32),
                affect=TextAffect(
                    emotions={l: round(float(v), 4) for l, v in zip(self.labels, p)},
                    active=active or [self.labels[int(p.argmax())]],
                    ekman=ekman,
                    valence=round(float(va[i, 0]), 4), arousal=round(float(va[i, 1]), 4),
                    risk=round(float(r_all[i]), 4), distortion=round(d, 4),
                    distortion_type=self.distortion_types[int(dtype[i].argmax())] if d >= self.distortion_threshold else None,
                ),
            ))
        return out


_sense: AuraSense | None = None
_lock = threading.Lock()


def get_sense() -> AuraSense:
    global _sense
    with _lock:
        if _sense is None:
            _sense = AuraSense()
        return _sense
