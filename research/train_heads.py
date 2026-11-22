"""Train the affect heads that sit on top of the shared MiniLM sentence encoder.

Three tasks, three model families each:
  * emotion    GoEmotions, 28-way multi-label
  * risk       Reddit SuicideWatch/depression vs. non-clinical posts, binary
  * distortion Shreevastava & Foltz therapist corpus, binary detection + 10-way type

  tfidf_lr    TF-IDF (1-2 grams) + logistic regression       -- lexical baseline
  minilm_lr   frozen MiniLM embedding + logistic regression  -- shared encoder, linear head
  minilm_mlp  frozen MiniLM embedding + 1-hidden-layer MLP   -- shared encoder, deployed head

Run:  python train_heads.py            (all tasks)
      python train_heads.py emotion    (one task)
"""

from __future__ import annotations

import base64
import json
import sys
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier

from common import (
    DISTORTIONS, EKMAN, EKMAN_LABELS, GOEMO_LABELS, RESULTS, SEED, embed,
    load_distortion, load_emobank, load_goemotions, load_risk, save_json, seed_everything,
)


# --------------------------------------------------------------------------- helpers


class MLP(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.drop = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(x))))


def train_mlp(x_tr, y_tr, x_va, y_va, *, multilabel: bool, n_out: int, hidden: int = 512,
              epochs: int = 60, lr: float = 2e-3, wd: float = 1e-4, pos_weight=None):
    """Small MLP trained with early stopping on validation loss."""
    seed_everything()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(x_tr.shape[1], hidden, n_out).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xt = torch.tensor(x_tr, device=dev)
    xv = torch.tensor(x_va, device=dev)
    if multilabel:
        yt = torch.tensor(y_tr, dtype=torch.float32, device=dev)
        yv = torch.tensor(y_va, dtype=torch.float32, device=dev)
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=dev) if pos_weight is not None else None
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        yt = torch.tensor(y_tr, dtype=torch.long, device=dev)
        yv = torch.tensor(y_va, dtype=torch.long, device=dev)
        loss_fn = torch.nn.CrossEntropyLoss()

    best, best_state, patience = float("inf"), None, 0
    n = len(xt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss_fn(model(xt[b]), yt[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(xv), yv).item()
        if vl < best - 1e-4:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_logits(model, x):
    dev = next(model.parameters()).device
    with torch.no_grad():
        return model(torch.tensor(x, device=dev)).cpu().numpy()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def tune_thresholds(y_true, probs):
    """Per-label threshold that maximises F1 on validation data."""
    grid = np.linspace(0.05, 0.9, 35)
    out = np.zeros(probs.shape[1], dtype=np.float32)
    for j in range(probs.shape[1]):
        scores = [f1_score(y_true[:, j], probs[:, j] >= t, zero_division=0) for t in grid]
        out[j] = grid[int(np.argmax(scores))]
    return out


def to_ekman(y):
    idx = {l: i for i, l in enumerate(GOEMO_LABELS)}
    out = np.zeros((len(y), len(EKMAN_LABELS)), dtype=np.int64)
    for k, members in enumerate(EKMAN.values()):
        out[:, k] = y[:, [idx[m] for m in members]].max(axis=1)
    return out


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode()


def export_mlp(model) -> dict:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {
        "type": "mlp",
        "activation": "gelu",
        "shape": [int(sd["fc1.weight"].shape[1]), int(sd["fc1.weight"].shape[0]), int(sd["fc2.weight"].shape[0])],
        "W1": b64(sd["fc1.weight"]), "b1": b64(sd["fc1.bias"]),
        "W2": b64(sd["fc2.weight"]), "b2": b64(sd["fc2.bias"]),
    }


def export_linear(W: np.ndarray, b: np.ndarray) -> dict:
    return {"type": "linear", "shape": [int(W.shape[1]), int(W.shape[0])], "W": b64(W), "b": b64(b)}


def fmt(d):
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating, int, np.integer)) else v) for k, v in d.items()}


# --------------------------------------------------------------------------- emotion


def run_emotion(report: dict, export: dict):
    print("\n== emotion (GoEmotions) ==")
    data = load_goemotions()
    (t_tr, y_tr), (t_va, y_va), (t_te, y_te) = data["train"], data["validation"], data["test"]
    results = {}

    def evaluate(name, p_va, p_te, secs):
        th = tune_thresholds(y_va, p_va)
        pred = (p_te >= th).astype(int)
        empty = pred.sum(1) == 0  # always emit at least one label
        pred[empty, p_te[empty].argmax(1)] = 1
        ek_true, ek_pred = to_ekman(y_te.astype(int)), to_ekman(pred)
        per_label = f1_score(y_te, pred, average=None, zero_division=0)
        results[name] = fmt({
            "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
            "micro_f1": f1_score(y_te, pred, average="micro", zero_division=0),
            "weighted_f1": f1_score(y_te, pred, average="weighted", zero_division=0),
            "ekman_macro_f1": f1_score(ek_true, ek_pred, average="macro", zero_division=0),
            "ekman_micro_f1": f1_score(ek_true, ek_pred, average="micro", zero_division=0),
            "train_seconds": secs,
        })
        results[name]["per_label_f1"] = {l: round(float(f), 4) for l, f in zip(GOEMO_LABELS, per_label)}
        print(f"  {name:11s} macroF1={results[name]['macro_f1']:.3f} microF1={results[name]['micro_f1']:.3f} "
              f"ekman macroF1={results[name]['ekman_macro_f1']:.3f}")
        return th, pred

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    X_tr = vec.fit_transform(t_tr)
    clf = OneVsRestClassifier(LogisticRegression(C=4.0, max_iter=2000), n_jobs=-1).fit(X_tr, y_tr)
    evaluate("tfidf_lr", clf.predict_proba(vec.transform(t_va)), clf.predict_proba(vec.transform(t_te)), time.time() - t0)

    e_tr, e_va, e_te = embed(t_tr, "goemo_tr"), embed(t_va, "goemo_va"), embed(t_te, "goemo_te")

    t0 = time.time()
    lr = OneVsRestClassifier(LogisticRegression(C=2.0, max_iter=3000), n_jobs=-1).fit(e_tr, y_tr)
    evaluate("minilm_lr", lr.predict_proba(e_va), lr.predict_proba(e_te), time.time() - t0)

    t0 = time.time()
    pos = y_tr.mean(0)
    pw = np.clip((1 - pos) / np.maximum(pos, 1e-4), 1.0, 10.0) ** 0.5
    mlp = train_mlp(e_tr, y_tr, e_va, y_va, multilabel=True, n_out=len(GOEMO_LABELS), pos_weight=pw)
    p_va, p_te = sigmoid(mlp_logits(mlp, e_va)), sigmoid(mlp_logits(mlp, e_te))
    th, pred = evaluate("minilm_mlp", p_va, p_te, time.time() - t0)

    np.save(RESULTS / "emotion_test_probs.npy", p_te)
    np.save(RESULTS / "emotion_test_true.npy", y_te)
    ek_cm = confusion_matrix(to_ekman(y_te.astype(int)).argmax(1), to_ekman(pred).argmax(1))
    report["emotion"] = {"dataset": "GoEmotions simplified", "n": {"train": len(t_tr), "val": len(t_va), "test": len(t_te)},
                         "models": results, "ekman_confusion_top1": ek_cm.tolist()}
    export["emotion"] = {"labels": GOEMO_LABELS, "thresholds": [round(float(x), 4) for x in th],
                         "ekman": EKMAN, "head": export_mlp(mlp)}


# --------------------------------------------------------------------------- risk


def threshold_for_recall(y, p, target):
    _, rec, th = precision_recall_curve(y, p)
    ok = np.flatnonzero(rec[:-1] >= target)
    return float(th[ok[-1]]) if len(ok) else float(th[0])


def run_risk(report: dict, export: dict):
    print("\n== risk (SuicideWatch) ==")
    data = load_risk()
    (t_tr, y_tr), (t_va, y_va), (t_te, y_te) = data["train"], data["validation"], data["test"]
    results = {}

    def evaluate(name, p_va, p_te, secs):
        t_r95 = threshold_for_recall(y_va, p_va, 0.95)
        out = {
            "auroc": roc_auc_score(y_te, p_te),
            "auprc": average_precision_score(y_te, p_te),
            "f1@0.5": f1_score(y_te, p_te >= 0.5),
            "precision@0.5": precision_score(y_te, p_te >= 0.5),
            "recall@0.5": recall_score(y_te, p_te >= 0.5),
            "threshold_r95": t_r95,
            "recall@r95": recall_score(y_te, p_te >= t_r95),
            "precision@r95": precision_score(y_te, p_te >= t_r95),
            "train_seconds": secs,
        }
        results[name] = fmt(out)
        print(f"  {name:11s} AUROC={out['auroc']:.4f} F1={out['f1@0.5']:.4f} P@R95={out['precision@r95']:.4f}")
        return t_r95

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=300_000)
    X_tr = vec.fit_transform(t_tr)
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(X_tr, y_tr)
    evaluate("tfidf_lr", clf.predict_proba(vec.transform(t_va))[:, 1], clf.predict_proba(vec.transform(t_te))[:, 1], time.time() - t0)

    e_tr, e_va, e_te = embed(t_tr, "risk_tr"), embed(t_va, "risk_va"), embed(t_te, "risk_te")

    t0 = time.time()
    lr = LogisticRegression(C=4.0, max_iter=3000).fit(e_tr, y_tr)
    evaluate("minilm_lr", lr.predict_proba(e_va)[:, 1], lr.predict_proba(e_te)[:, 1], time.time() - t0)

    t0 = time.time()
    mlp = train_mlp(e_tr, y_tr, e_va, y_va, multilabel=False, n_out=2)
    p_va, p_te = softmax(mlp_logits(mlp, e_va))[:, 1], softmax(mlp_logits(mlp, e_te))[:, 1]
    t95 = evaluate("minilm_mlp", p_va, p_te, time.time() - t0)
    t80 = threshold_for_recall(y_va, p_va, 0.80)

    np.save(RESULTS / "risk_test_probs.npy", p_te)
    np.save(RESULTS / "risk_test_true.npy", y_te)
    report["risk"] = {"dataset": "Komati (2021) SuicideWatch/depression vs r/teenagers, stratified subsample",
                      "n": {"train": len(t_tr), "val": len(t_va), "test": len(t_te)}, "models": results}
    export["risk"] = {"labels": ["none", "risk"], "threshold_elevated": round(t95, 4),
                      "threshold_crisis": round(t80, 4), "head": export_mlp(mlp)}


# --------------------------------------------------------------------------- distortion


def run_distortion(report: dict, export: dict):
    print("\n== distortion (Shreevastava & Foltz) ==")
    data = load_distortion()
    (t_tr, b_tr, k_tr), (t_va, b_va, _), (t_te, b_te, k_te) = data["train"], data["validation"], data["test"]
    results = {"detection": {}, "type": {}}

    e_tr, e_va, e_te = embed(t_tr, "dist_tr"), embed(t_va, "dist_va"), embed(t_te, "dist_te")

    def det_eval(name, p_va, p_te):
        grid = np.linspace(0.2, 0.8, 25)
        t = grid[int(np.argmax([f1_score(b_va, p_va >= g) for g in grid]))]
        results["detection"][name] = fmt({
            "f1": f1_score(b_te, p_te >= t), "precision": precision_score(b_te, p_te >= t),
            "recall": recall_score(b_te, p_te >= t), "auroc": roc_auc_score(b_te, p_te),
            "threshold": t, "all_positive_f1": f1_score(b_te, np.ones_like(b_te)),
        })
        r = results["detection"][name]
        print(f"  detect {name:11s} F1={r['f1']:.3f} AUROC={r['auroc']:.3f}")
        return t

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X_tr = vec.fit_transform(t_tr)
    c = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced").fit(X_tr, b_tr)
    det_eval("tfidf_lr", c.predict_proba(vec.transform(t_va))[:, 1], c.predict_proba(vec.transform(t_te))[:, 1])

    c = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(e_tr, b_tr)
    det_threshold = det_eval("minilm_lr", c.predict_proba(e_va)[:, 1], c.predict_proba(e_te)[:, 1])
    # softmax over [-z/2, z/2] equals sigmoid(z), so a 2-way linear head reproduces the binary LR exactly
    det_head = export_linear(np.vstack([-c.coef_[0], c.coef_[0]]) / 2, np.array([-c.intercept_[0], c.intercept_[0]]) / 2)

    m_tr, m_te = k_tr >= 0, k_te >= 0
    type_head = None
    for name, (Xa, Xb) in {"tfidf_lr": (vec.transform(t_tr), vec.transform(t_te)), "minilm_lr": (e_tr, e_te)}.items():
        c = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(Xa[m_tr], k_tr[m_tr])
        pred = c.predict(Xb[m_te])
        results["type"][name] = fmt({
            "macro_f1": f1_score(k_te[m_te], pred, average="macro"),
            "accuracy": float((pred == k_te[m_te]).mean()),
            "chance_accuracy": 1 / len(DISTORTIONS),
        })
        r = results["type"][name]
        print(f"  type   {name:11s} macroF1={r['macro_f1']:.3f} acc={r['accuracy']:.3f}")
        if name == "minilm_lr":
            type_head = export_linear(c.coef_, c.intercept_)

    report["distortion"] = {"dataset": "Shreevastava & Foltz (2021)",
                            "n": {"train": len(t_tr), "val": len(t_va), "test": len(t_te)}, "models": results}
    export["distortion"] = {"threshold": round(float(det_threshold), 4), "head": det_head,
                            "types": DISTORTIONS, "type_head": type_head}


# --------------------------------------------------------------------------- main


def run_va(report: dict, export: dict):
    print("\n== valence/arousal (EmoBank) ==")
    from scipy.stats import pearsonr
    from sklearn.linear_model import RidgeCV

    d = load_emobank()
    (t_tr, y_tr), (t_te, y_te) = d["train"], d["test"]
    results = {}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    feats = {
        "tfidf_ridge": (vec.fit_transform(t_tr), lambda: vec.transform(t_te)),
        "minilm_ridge": (embed(t_tr, "emob_tr"), lambda: embed(t_te, "emob_te")),
    }
    for name, (Xa, make_test) in feats.items():
        reg = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(Xa, y_tr)
        pred = reg.predict(make_test())
        results[name] = {"pearson_valence": round(float(pearsonr(pred[:, 0], y_te[:, 0])[0]), 4),
                         "pearson_arousal": round(float(pearsonr(pred[:, 1], y_te[:, 1])[0]), 4)}
        print(f"  {name:12s} r_V={results[name]['pearson_valence']:.3f} r_A={results[name]['pearson_arousal']:.3f}")
    report["va"] = {"dataset": "EmoBank", "n": {"train": len(t_tr), "test": len(t_te)}, "models": results}


def main():
    seed_everything()
    tasks = sys.argv[1:] or ["emotion", "risk", "distortion", "va"]
    # Frozen-encoder heads are the ablation baseline for Aura-Sense. The deployed model
    # is exported by train_aura_sense.py.
    report_path, export_path = RESULTS / "heads_report.json", RESULTS / "frozen_heads.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    export = json.loads(export_path.read_text()) if export_path.exists() else {}
    export.update({"encoder": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384, "seed": SEED})
    runners = {"emotion": run_emotion, "risk": run_risk, "distortion": run_distortion, "va": run_va}
    for t in tasks:
        runners[t](report, export)
        save_json(report, report_path)
        export_path.write_text(json.dumps(export), encoding="utf-8")
    print(f"\nreport -> {report_path}\nheads  -> {export_path} ({export_path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
