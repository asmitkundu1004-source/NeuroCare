"""Aura-Face: facial affect from 52 blendshape coefficients, never from pixels.

Design
  * Input is the MediaPipe blendshape vector (the browser computes the same one), so the
    model cannot see identity, skin, background or anything else in the frame.
  * Bilateral encoding: each left/right blendshape pair is re-expressed as its mean and
    its absolute asymmetry. Some expressions are defined by asymmetry (contempt is a
    one-sided lip-corner raise, FACS AU12/AU14 unilateral), which a plain vector hides.
  * Mirror augmentation: swapping left and right never changes the expression label,
    so training samples are mirrored at random.
  * Shared affect space: class probabilities are projected through the circumplex that
    Aura-Sense learned from text, so face and text report valence/arousal in one space.

Baselines: a zero-shot FACS rule model, multinomial logistic regression and gradient
boosting on the raw 52 features.

Run: python face/train_aura_face.py      (after extract_blendshapes.py)
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import CACHE, RESULTS, ROOT, SEED, save_json, seed_everything  # noqa: E402

WEB_MODELS = ROOT.parent / "web" / "public" / "models"
ENSEMBLE = 5
# face label -> GoEmotions label whose learned circumplex position it borrows
TO_GOEMO = {"anger": "anger", "surprise": "surprise", "contempt": "disapproval", "happy": "joy",
            "neutral": "neutral", "fear": "fear", "sad": "sadness", "disgust": "disgust"}


# --------------------------------------------------------------------------- features


def pairs(names: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    """Indices of unpaired blendshapes and (left, right) index pairs."""
    idx = {n: i for i, n in enumerate(names)}
    lr, single = [], []
    for n, i in idx.items():
        if n.endswith("Left"):
            r = n[:-4] + "Right"
            if r in idx:
                lr.append((i, idx[r]))
        elif not n.endswith("Right"):
            single.append(i)
    return single, lr


def bilateral(x: np.ndarray, single, lr) -> np.ndarray:
    left = x[:, [l for l, _ in lr]]
    right = x[:, [r for _, r in lr]]
    return np.hstack([x[:, single], (left + right) / 2, np.abs(left - right)]).astype(np.float32)


def mirror(x: np.ndarray, lr) -> np.ndarray:
    m = x.copy()
    for l, r in lr:
        m[:, [l, r]] = x[:, [r, l]]
    return m


# --------------------------------------------------------------------------- FACS rules


def facs_scores(x: np.ndarray, names: list[str], labels: list[str]) -> np.ndarray:
    """Zero-shot prototype scores from FACS action-unit correspondences (Ekman & Friesen, 1978)."""
    b = {n: x[:, i] for i, n in enumerate(names)}

    def avg(*ks):
        return np.mean([b[k] for k in ks], axis=0)

    smile = avg("mouthSmileLeft", "mouthSmileRight")
    s = {
        "happy": 0.7 * smile + 0.3 * avg("cheekSquintLeft", "cheekSquintRight"),
        "sad": 0.5 * b["browInnerUp"] + 0.5 * avg("mouthFrownLeft", "mouthFrownRight"),
        "surprise": 0.35 * b["browInnerUp"] + 0.25 * avg("browOuterUpLeft", "browOuterUpRight")
                    + 0.25 * b["jawOpen"] + 0.15 * avg("eyeWideLeft", "eyeWideRight"),
        "fear": 0.4 * b["browInnerUp"] + 0.35 * avg("eyeWideLeft", "eyeWideRight")
                + 0.25 * avg("mouthStretchLeft", "mouthStretchRight"),
        "anger": 0.5 * avg("browDownLeft", "browDownRight") + 0.25 * avg("eyeSquintLeft", "eyeSquintRight")
                 + 0.25 * avg("mouthPressLeft", "mouthPressRight"),
        "disgust": 0.6 * avg("noseSneerLeft", "noseSneerRight") + 0.4 * avg("mouthUpperUpLeft", "mouthUpperUpRight"),
        "contempt": np.abs(b["mouthSmileLeft"] - b["mouthSmileRight"])
                    + 0.5 * np.abs(b["mouthDimpleLeft"] - b["mouthDimpleRight"]),
    }
    peak = np.max(np.stack(list(s.values())), axis=0)
    s["neutral"] = np.clip(0.35 - peak, 0, None) + 0.05
    return np.stack([s[l] for l in labels], axis=1)


# --------------------------------------------------------------------------- model


class AuraFace(torch.nn.Module):
    def __init__(self, d_in, n_out, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(x_tr, y_tr, x_va, y_va, lr_pairs, single, n_out, *, augment=True, bilateral_feats=True,
              epochs=300, seed=SEED):
    seed_everything(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def feat(x):
        return bilateral(x, single, lr_pairs) if bilateral_feats else x.astype(np.float32)

    f_tr = feat(x_tr)
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6

    def prep(x):
        return ((feat(x) - mu) / sd).astype(np.float32)

    model = AuraFace(f_tr.shape[1], n_out).to(dev)
    counts = np.bincount(y_tr, minlength=n_out)
    w = torch.tensor((counts.sum() / (n_out * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32, device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xv = torch.tensor(prep(x_va), device=dev)
    yt = torch.tensor(y_tr, device=dev)
    best, best_state, patience = -1.0, None, 0
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        xa = x_tr
        if augment:
            flip = rng.random(len(x_tr)) < 0.5
            xa = np.where(flip[:, None], mirror(x_tr, lr_pairs), x_tr)
        xt = torch.tensor(prep(xa), device=dev)
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).cpu().numpy()
        f = f1_score(y_va, pred, average="macro")
        if f > best:
            best, patience = f, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 25:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, prep, mu, sd


def predict(model, prep, x):
    dev = next(model.parameters()).device
    with torch.no_grad():
        return torch.softmax(model(torch.tensor(prep(x), device=dev)), 1).cpu().numpy()


def metrics(y, p, labels):
    pred = p.argmax(1)
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "per_class_f1": dict(zip(labels, f1_score(y, pred, average=None).round(4).tolist())),
    }


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def main():
    tr, te = np.load(CACHE / "affectnet_blendshapes_train.npz"), np.load(CACHE / "affectnet_blendshapes_val.npz")
    names, labels = [str(n) for n in tr["names"]], [str(l) for l in tr["labels"]]
    x_all, y_all = tr["x"][tr["found"]], tr["y"][tr["found"]]
    x_te, y_te = te["x"][te["found"]], te["y"][te["found"]]
    x_tr, x_va, y_tr, y_va = train_test_split(x_all, y_all, test_size=0.1, random_state=SEED, stratify=y_all)
    single, lr_pairs = pairs(names)
    print(f"faces: train {len(x_tr)} val {len(x_va)} test {len(x_te)} | detection rate "
          f"train {tr['found'].mean():.3f} test {te['found'].mean():.3f} | {len(lr_pairs)} L/R pairs")

    report = {"dataset": "AffectNet (8-class subset, HF Mauregato/affectnet_short)", "labels": labels,
              "n": {"train": int(len(x_tr)), "val": int(len(x_va)), "test": int(len(x_te))},
              "face_detection_rate": {"train": round(float(tr["found"].mean()), 4), "test": round(float(te["found"].mean()), 4)},
              "class_counts_test": np.bincount(y_te, minlength=len(labels)).tolist(), "models": {}}

    report["models"]["facs_rules_zero_shot"] = metrics(y_te, facs_scores(x_te, names, labels), labels)
    lr = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced").fit(x_tr, y_tr)
    report["models"]["logreg_raw52"] = metrics(y_te, lr.predict_proba(x_te), labels)
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, class_weight="balanced",
                                        random_state=SEED).fit(x_tr, y_tr)
    report["models"]["gboost_raw52"] = metrics(y_te, gb.predict_proba(x_te), labels)

    ablations = {
        "mlp_raw52": dict(augment=False, bilateral_feats=False),
        "mlp_bilateral": dict(augment=False, bilateral_feats=True),
        "mlp_mirror": dict(augment=True, bilateral_feats=False),
        "aura_face": dict(augment=True, bilateral_feats=True),
    }
    for name, kw in ablations.items():
        model, prep, mu, sd = train_mlp(x_tr, y_tr, x_va, y_va, lr_pairs, single, len(labels), **kw)
        report["models"][name] = metrics(y_te, predict(model, prep, x_te), labels)

    # deployed model: a 5-seed ensemble of the full design (the gain comes from ensembling)
    members = [train_mlp(x_tr, y_tr, x_va, y_va, lr_pairs, single, len(labels), seed=s) for s in range(ENSEMBLE)]
    p = np.mean([predict(m, prep, x_te) for m, prep, _, _ in members], axis=0)
    report["models"]["aura_face_ensemble"] = metrics(y_te, p, labels)
    model, prep, mu, sd = members[0]
    for k, v in report["models"].items():
        print(f"  {k:22s} acc={v['accuracy']:.3f} bal_acc={v['balanced_accuracy']:.3f} macroF1={v['macro_f1']:.3f}")

    report["aura_face_confusion"] = confusion_matrix(y_te, p.argmax(1)).tolist()
    n_params = sum(t.numel() for t in model.parameters()) * ENSEMBLE
    cpu = AuraFace(len(mu), len(labels))
    cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu.eval()
    xb = torch.tensor(prep(x_te[:1]))
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(2000):
            cpu(xb)
    report["aura_face_size"] = {"parameters": int(n_params), "bytes_fp32": int(n_params * 4),
                                "cpu_us_per_frame": round((time.perf_counter() - t0) / 2000 * 1e6, 1)}
    save_json(report, RESULTS / "aura_face_report.json")

    def layers_of(m):
        sd_ = {k: v.detach().cpu().numpy() for k, v in m.state_dict().items()}
        return [{"W": b64(sd_[f"net.{i}.weight"]), "b": b64(sd_[f"net.{i}.bias"]),
                 "shape": list(sd_[f"net.{i}.weight"].shape)} for i in (0, 3, 6)]

    export = {"name": "Aura-Face", "version": "1.0.0", "labels": labels, "blendshapes": names,
              "single": single, "pairs": lr_pairs, "mean": b64(mu), "std": b64(sd),
              "members": [layers_of(m) for m, *_ in members], "activation": "gelu", "to_goemotions": TO_GOEMO,
              "metrics": {k: v for k, v in report["models"]["aura_face_ensemble"].items() if k != "per_class_f1"}}
    WEB_MODELS.mkdir(parents=True, exist_ok=True)
    (WEB_MODELS / "aura-face.json").write_text(json.dumps(export), encoding="utf-8")
    print(f"exported -> {WEB_MODELS / 'aura-face.json'} ({n_params} params)")


if __name__ == "__main__":
    main()
