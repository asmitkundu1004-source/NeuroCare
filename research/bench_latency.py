"""CPU latency and footprint of the on-device pipeline.

Measured on whatever machine runs it; the paper reports the machine alongside.

Run: python bench_latency.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from neurocare.agem import AgemParams, rank  # noqa: E402

RESULTS = ROOT / "results"
MODEL_DIR = ROOT.parent / "backend" / "models" / "aura-sense"


def stats(xs):
    xs = np.array(xs) * 1000
    return {"median_ms": round(float(np.median(xs)), 2), "p90_ms": round(float(np.percentile(xs, 90)), 2)}


def bench_onnx(path: Path, texts, threads: int):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    tok.enable_truncation(max_length=128)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    feeds = []
    for t in texts:
        e = tok.encode(t)
        feeds.append({"input_ids": np.array([e.ids], np.int64), "attention_mask": np.array([e.attention_mask], np.int64)})
    for f in feeds[:10]:
        sess.run(None, f)
    times = []
    for f in feeds:
        t0 = time.perf_counter()
        sess.run(None, f)
        times.append(time.perf_counter() - t0)
    return stats(times), float(np.mean([f["input_ids"].shape[1] for f in feeds]))


def main():
    from common import load_goemotions

    texts = load_goemotions()["test"][0][:200]
    report = {"machine": {"cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
                          "os": f"{platform.system()} {platform.release()}"},
              "n_sentences": len(texts), "aura_sense": {}}
    q8 = MODEL_DIR / "model.onnx"
    fp32 = ROOT / "checkpoints" / "aura_sense_fp32.onnx"
    for name, path in (("weight_only_8bit", q8), ("fp32", fp32)):
        if not path.exists():
            continue
        t1, mean_tokens = bench_onnx(path, texts, 1)
        t4, _ = bench_onnx(path, texts, 4)
        report["mean_tokens"] = round(mean_tokens, 1)
        report["aura_sense"][name] = {"size_mb": round(path.stat().st_size / 1e6, 2), "threads_1": t1, "threads_4": t4}
        print(name, report["aura_sense"][name])

    # one shared pass per turn, against the four separate encoders a conventional design would run
    single = report["aura_sense"]["weight_only_8bit"]["threads_4"]["median_ms"]
    report["shared_encoder"] = {"passes_per_turn": 1, "conventional_passes_per_turn": 4,
                                "estimated_conventional_ms": round(4 * single, 2), "saving_ms": round(3 * single, 2)}

    rng = np.random.default_rng(0)
    report["agem"] = {}
    for n in (100, 1000, 10000):
        mem = rng.normal(size=(n, 384)).astype(np.float32)
        mem /= np.linalg.norm(mem, axis=1, keepdims=True)
        q = mem[0] + 0.1 * rng.normal(size=384).astype(np.float32)
        q /= np.linalg.norm(q)
        v = rng.uniform(-1, 1, n).astype(np.float32)
        a = rng.uniform(-1, 1, n).astype(np.float32)
        age, last = rng.uniform(0, 90, n), np.full(n, 10 ** 6)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            rank(q, mem, v, a, age, last, 0.1, AgemParams())
            times.append(time.perf_counter() - t0)
        report["agem"][str(n)] = stats(times)
        print("agem", n, report["agem"][str(n)])

    face = json.loads((RESULTS / "aura_face_report.json").read_text())
    report["aura_face"] = {"parameters_total": face["aura_face_size"]["parameters"],
                           "cpu_us_per_frame_per_member": face["aura_face_size"]["cpu_us_per_frame"]}
    (RESULTS / "latency_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
