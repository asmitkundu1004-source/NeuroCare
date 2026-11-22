"""Upper-bound comparison: fine-tune MiniLM end-to-end, one model per task.

The deployed system shares ONE frozen encoder across memory retrieval and all affect
heads. This script measures what that sharing costs in accuracy relative to giving
each task its own fine-tuned copy of the encoder (3x the parameters, 3x the passes).

Run:  python finetune.py emotion | risk
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from transformers import AutoModel, AutoTokenizer

from common import ENCODER, RESULTS, load_goemotions, load_risk, save_json, seed_everything
from train_heads import to_ekman, tune_thresholds


class Classifier(torch.nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ENCODER)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, n_out)

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        return self.head(pooled)


def batches(tok, texts, labels, bs, max_len, shuffle):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        enc = tok([texts[j] for j in b], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        yield enc["input_ids"], enc["attention_mask"], labels[b]


def predict(model, tok, texts, max_len, dev, multilabel):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
        for ids, mask, _ in batches(tok, texts, np.zeros(len(texts)), 128, max_len, False):
            logits = model(ids.to(dev), mask.to(dev)).float()
            out.append(torch.sigmoid(logits) if multilabel else torch.softmax(logits, -1)[:, 1])
    return torch.cat(out).cpu().numpy()


def train(task: str):
    seed_everything()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if task == "emotion":
        d = load_goemotions()
        multilabel, n_out, max_len, epochs, bs = True, 28, 64, 3, 64
    else:
        d = load_risk()
        multilabel, n_out, max_len, epochs, bs = False, 2, 256, 2, 32
    (t_tr, y_tr), (t_va, y_va), (t_te, y_te) = d["train"], d["validation"], d["test"]

    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = Classifier(n_out).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    steps = epochs * ((len(t_tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss() if multilabel else torch.nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total, n = 0.0, 0
        for ids, mask, y in batches(tok, t_tr, y_tr, bs, max_len, True):
            y = torch.tensor(y, dtype=torch.float32 if multilabel else torch.long, device=dev)
            opt.zero_grad()
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=dev == "cuda"):
                loss = loss_fn(model(ids.to(dev), mask.to(dev)).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item(), n + 1
        print(f"  epoch {ep + 1}/{epochs} loss={total / n:.4f} ({time.time() - t0:.0f}s)")
    secs = time.time() - t0

    p_va = predict(model, tok, t_va, max_len, dev, multilabel)
    p_te = predict(model, tok, t_te, max_len, dev, multilabel)
    if multilabel:
        th = tune_thresholds(y_va, p_va)
        pred = (p_te >= th).astype(int)
        empty = pred.sum(1) == 0
        pred[empty, p_te[empty].argmax(1)] = 1
        res = {
            "macro_f1": f1_score(y_te, pred, average="macro", zero_division=0),
            "micro_f1": f1_score(y_te, pred, average="micro", zero_division=0),
            "ekman_macro_f1": f1_score(to_ekman(y_te.astype(int)), to_ekman(pred), average="macro", zero_division=0),
        }
    else:
        res = {"auroc": roc_auc_score(y_te, p_te), "auprc": average_precision_score(y_te, p_te),
               "f1@0.5": f1_score(y_te, p_te >= 0.5)}
    res = {k: round(float(v), 4) for k, v in res.items()}
    res.update({"train_seconds": round(secs, 1), "epochs": epochs,
                "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"})
    print(f"  {task} finetuned: {res}")

    path = RESULTS / "finetune_report.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report[task] = res
    save_json(report, path)


if __name__ == "__main__":
    for t in sys.argv[1:] or ["emotion", "risk"]:
        train(t)
