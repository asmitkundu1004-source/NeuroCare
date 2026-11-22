"""Shared data loading, embedding cache and label definitions for the NeuroCare experiments."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
EXPORT = ROOT.parent / "backend" / "models"

for _p in (DATA, CACHE, RESULTS, FIGURES, EXPORT):
    _p.mkdir(parents=True, exist_ok=True)

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 13

GOEMO_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]

# Ekman grouping published with GoEmotions (Demszky et al., 2020).
EKMAN = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}
EKMAN_LABELS = list(EKMAN)

DISTORTIONS = [
    "All-or-nothing thinking", "Emotional Reasoning", "Fortune-telling", "Labeling",
    "Magnification", "Mental filter", "Mind Reading", "Overgeneralization",
    "Personalization", "Should statements",
]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- datasets


def load_goemotions():
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    out = {}
    for split in ("train", "validation", "test"):
        texts = list(ds[split]["text"])
        y = np.zeros((len(texts), len(GOEMO_LABELS)), dtype=np.float32)
        for i, labels in enumerate(ds[split]["labels"]):
            y[i, labels] = 1.0
        out[split] = (texts, y)
    return out


def load_risk(n_train: int = 30000, n_val: int = 5000, n_test: int = 10000):
    """Reddit SuicideWatch/depression vs. r/teenagers (Komati, 2021; HF mirror jquiros/suicide)."""
    from datasets import load_dataset

    ds = load_dataset("jquiros/suicide", split="train")
    texts = [t if isinstance(t, str) else "" for t in ds["text"]]
    labels = np.array([1 if c == "suicide" else 0 for c in ds["class"]], dtype=np.int64)
    keep = np.array([len(t.strip()) > 0 for t in texts])
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    total = n_train + n_val + n_test
    idx = idx[:total]
    parts = np.split(idx, [n_train, n_train + n_val])
    names = ("train", "validation", "test")
    return {n: ([texts[i] for i in p], labels[p]) for n, p in zip(names, parts)}


def load_distortion():
    """Shreevastava & Foltz (2021) therapist Q&A corpus with dominant distortion annotations."""
    from datasets import load_dataset

    ds = load_dataset("danthareja/cognitive-distortion")
    names = ds["train"].features["dominant_distortion"].names
    out = {}
    for split in ("train", "test"):
        texts = [str(t).strip() for t in ds[split][" patient_question"]]
        raw = [names[i] for i in ds[split]["dominant_distortion"]]
        binary = np.array([0 if r == "No Distortion" else 1 for r in raw], dtype=np.int64)
        kind = np.array([DISTORTIONS.index(r) if r in DISTORTIONS else -1 for r in raw])
        out[split] = (texts, binary, kind)
    # carve a validation split from train, stratified on the binary label
    from sklearn.model_selection import train_test_split

    t, b, k = out["train"]
    tr, va = train_test_split(np.arange(len(t)), test_size=0.15, random_state=SEED, stratify=b)
    out["validation"] = ([t[i] for i in va], b[va], k[va])
    out["train"] = ([t[i] for i in tr], b[tr], k[tr])
    return out


def load_emobank():
    """EmoBank (Buechel & Hahn, 2017): 10k sentences with mean reader V/A ratings on a 1-5 scale.

    Ratings are mapped by (x - 3) / 2 so they share a scale with the circumplex anchors.
    Pearson r, the reported metric, is unaffected by this.
    """
    import pandas as pd

    df = pd.read_csv(DATA / "emobank.csv").dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    out = {}
    for split, name in (("train", "train"), ("dev", "validation"), ("test", "test")):
        d = df[df["split"] == split]
        out[name] = (d["text"].tolist(), ((d[["V", "A"]].to_numpy(np.float32) - 3.0) / 2.0))
    return out


def load_stsb():
    """STS-Benchmark, used only to check that fine-tuning preserves the retrieval geometry."""
    from datasets import load_dataset

    ds = load_dataset("sentence-transformers/stsb")
    return {s: (list(ds[s]["sentence1"]), list(ds[s]["sentence2"]), np.array(ds[s]["score"], np.float32))
            for s in ("validation", "test")}


def load_nrc_vad() -> dict[str, tuple[float, float]] | None:
    """NRC VAD lexicon (Mohammad, 2018) human norms, rescaled to [-1, 1]. None if absent."""
    for path in sorted((DATA / "nrc").rglob("*.txt")):
        name = path.name.lower()
        if "vad" not in name or "readme" in name or "sorted" in name:
            continue
        lex = {}
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                v, a = float(parts[1]), float(parts[2])
            except ValueError:
                continue
            lex[parts[0].lower()] = (v, a)
        if lex:
            vals = np.array(list(lex.values()))
            if vals.min() >= 0:  # v1 ships [0, 1]; v2 ships [-1, 1]
                lex = {k: (2 * v - 1, 2 * a - 1) for k, (v, a) in lex.items()}
            return lex
    return None


# --------------------------------------------------------------------------- embeddings

_model = None


def encoder():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(ENCODER, device=device)
        _model.max_seq_length = 256
    return _model


def embed(texts: list[str], tag: str, batch_size: int = 128) -> np.ndarray:
    """L2-normalised MiniLM sentence embeddings, cached on disk by content hash."""
    h = hashlib.sha1(("\x1f".join(texts)).encode("utf-8", "ignore")).hexdigest()[:16]
    path = CACHE / f"{tag}_{h}.npy"
    if path.exists():
        return np.load(path)
    vecs = encoder().encode(
        texts, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    np.save(path, vecs)
    return vecs
