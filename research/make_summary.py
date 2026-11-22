"""Collect every result file into one summary, the paper figures and Markdown tables.

Outputs
  results/summary.json     read by the website (landing page and Model Lab)
  results/tables.md        tables quoted in the paper and the project report
  figures/*.pdf|png        figures for the paper and the report

Run after the experiments: python make_summary.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


def pm(vals, d=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.{d}f}"
    return f"{np.mean(vals):.{d}f} ± {np.std(vals, ddof=1):.{d}f}"


def get(d, *path):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


# --------------------------------------------------------------------------- loaders


def aura_runs():
    runs = {}
    for f in sorted(glob.glob(str(RESULTS / "aura_sense_*_s*.json"))):
        r = json.loads(Path(f).read_text())
        runs.setdefault(r["config"], []).append(r)
    return runs


def teacher_stsb():
    cache = RESULTS / "stsb_teacher.json"
    if cache.exists():
        return json.loads(cache.read_text())["spearman"]
    from scipy.stats import spearmanr

    from common import embed, load_stsb

    s1, s2, gold = load_stsb()["test"]
    a, b = embed(s1, "stsb_s1"), embed(s2, "stsb_s2")
    rho = float(spearmanr((a * b).sum(1), gold).correlation)
    cache.write_text(json.dumps({"spearman": rho}))
    return rho


# --------------------------------------------------------------------------- tables


def text_table(runs, heads, ft, teacher):
    cols = ["Model", "Encoder passes / turn", "Emotion macro-F1", "Ekman macro-F1", "Valence r", "Arousal r",
            "Risk AUROC", "Distortion F1", "STS-B ρ"]
    rows, hl = [], []

    def hm(task, m, k):
        return get(heads, task, "models", m, k)

    if heads:
        rows.append(["TF-IDF + logistic / ridge (per task)", "—", fmt(hm("emotion", "tfidf_lr", "macro_f1")), fmt(hm("emotion", "tfidf_lr", "ekman_macro_f1")),
                     fmt(hm("va", "tfidf_ridge", "pearson_valence")), fmt(hm("va", "tfidf_ridge", "pearson_arousal")),
                     fmt(hm("risk", "tfidf_lr", "auroc")), fmt(get(heads, "distortion", "models", "detection", "tfidf_lr", "f1")), "—"])
        rows.append(["Frozen MiniLM + task heads", "1", fmt(hm("emotion", "minilm_mlp", "macro_f1")), fmt(hm("emotion", "minilm_mlp", "ekman_macro_f1")),
                     fmt(hm("va", "minilm_ridge", "pearson_valence")), fmt(hm("va", "minilm_ridge", "pearson_arousal")),
                     fmt(hm("risk", "minilm_mlp", "auroc")), fmt(get(heads, "distortion", "models", "detection", "minilm_lr", "f1")), fmt(teacher)])
    if ft:
        rows.append(["MiniLM fine-tuned separately per task", "one per task", fmt(get(ft, "emotion", "macro_f1")), fmt(get(ft, "emotion", "ekman_macro_f1")),
                     "—", "—", fmt(get(ft, "risk", "auroc")), "—", "—"])
    names = {"mt": "Multi-task fine-tune", "mt_distill": "+ retrieval-preserving distillation",
             "circ_uw": "+ circumplex, uncertainty-weighted", "aura_sense": "Aura-Sense (ours): + circumplex, fixed weight"}
    for cfg in ("mt", "mt_distill", "circ_uw", "aura_sense"):
        rs = runs.get(cfg)
        if not rs:
            continue
        seeds = f" (n={len(rs)} seeds)" if len(rs) > 1 else ""
        circ = cfg in ("circ_uw", "aura_sense")

        def cell(*p):
            return pm([get(r, *p) for r in rs])

        rows.append([names[cfg] + seeds, "1", cell("emotion", "macro_f1"), cell("emotion", "ekman_macro_f1"),
                     cell("va", "circumplex_pearson_valence" if circ else "pearson_valence"),
                     cell("va", "circumplex_pearson_arousal" if circ else "pearson_arousal"),
                     cell("risk", "auroc"), cell("distortion", "f1"), cell("retrieval", "stsb_spearman")])
        if cfg == "aura_sense":
            hl.append(len(rows) - 1)
    return {"id": "text", "title": "Text affect: baselines, ablations and Aura-Sense",
            "caption": "Held-out test sets: GoEmotions (5,427), EmoBank (1,000), Reddit risk (10,000), Shreevastava–Foltz (506), STS-B (1,379). "
                       "For circumplex models, valence and arousal are read through the learned circumplex.",
            "columns": cols, "rows": rows, "highlight": hl}


def face_table(face):
    names = {"facs_rules_zero_shot": "FACS prototype rules (zero-shot)", "logreg_raw52": "Logistic regression, 52 coefficients",
             "gboost_raw52": "Gradient boosting, 52 coefficients", "mlp_raw52": "MLP, 52 coefficients",
             "mlp_bilateral": "MLP + bilateral encoding", "mlp_mirror": "MLP + mirror augmentation",
             "aura_face": "MLP + both", "aura_face_ensemble": "Aura-Face: 5-member ensemble (deployed)"}
    rows, hl = [], []
    for k, n in names.items():
        m = face["models"].get(k)
        if not m:
            continue
        rows.append([n, fmt(m["accuracy"]), fmt(m["balanced_accuracy"]), fmt(m["macro_f1"])])
        if k == "aura_face_ensemble":
            hl.append(len(rows) - 1)
    return {"id": "face", "title": "Facial affect from blendshapes only (AffectNet, 8 classes)",
            "caption": f"Test n = {face['n']['test']:,}, chance accuracy 0.125. Faces detected in {face['face_detection_rate']['test'] * 100:.1f}% "
                       "of images. No model sees pixels.",
            "columns": ["Model", "Accuracy", "Balanced acc.", "Macro-F1"], "rows": rows, "highlight": hl}


def agem_table(agem):
    rows, hl = [], []
    for name, m in agem["methods"].items():
        rows.append([name, fmt(m["recall@3"], 2), fmt(m["isr"], 2), fmt(m["tangential_target_surfaced"], 2),
                     fmt(m["empty_rate_unrelated"], 2), fmt(m["redundancy"], 2)])
        if name == "AGEM (full)":
            hl.append(len(rows) - 1)
    n = agem["n"]
    return {"id": "agem", "title": "Memory policies on a scripted personal history",
            "caption": f"{n['memories']} memories ({n['painful']} painful), {n['relevant_queries']} relevant, {n['unrelated_queries']} unrelated and "
                       f"{n['tangential_queries']} tangential queries, {n['follow_up_turns']} follow-up turns. ISR: share of unrelated or tangential "
                       "queries where a painful memory was injected (lower is better).",
            "columns": ["Policy", "Recall@3 ↑", "ISR ↓", "Tangential pain ↓", "Silent when unrelated ↑", "Redundancy ↓"],
            "rows": rows, "highlight": hl}


def safety_table(safety, v1):
    rows, hl = [], []
    for label, rep in (("v1", v1), ("v2", safety)):
        if not rep or "heldout" not in rep:
            continue
        for mode in ("lexicon", "classifier", "hybrid"):
            d, h = rep["systems"][mode], rep["heldout"]["systems"][mode]
            rows.append([f"{mode} · lexicon {label}", fmt(d["explicit_crisis_recall"], 2), fmt(d["risk_alert_recall"], 2),
                         fmt(d["benign_false_alert"], 2), fmt(h["explicit_crisis_recall"], 2), fmt(h["risk_alert_recall"], 2),
                         fmt(h["idiom_false_alert"], 2)])
            if label == "v2" and mode == "hybrid":
                hl.append(len(rows) - 1)
    n, nh = safety["n"], safety["heldout"]["n"]
    return {"id": "safety", "title": "Safety gate on short conversational messages",
            "caption": f"Development probe: {n['explicit']} explicit, {n['indirect']} indirect, {n['distress']} distress, {n['idiom'] + n['everyday']} benign. "
                       f"Held-out probe written before lexicon v2: {nh['explicit']} explicit, {nh['indirect']} indirect, {nh['idiom']} idioms. "
                       "No configuration escalated ordinary distress to a crisis.",
            "columns": ["System", "Dev crisis recall", "Dev alert recall", "Dev benign false alert", "Held-out crisis recall",
                        "Held-out alert recall", "Held-out idiom false alert"], "rows": rows, "highlight": hl}


def runtime_table(parity, lat):
    rows = []
    for k, label in (("fp32", "fp32"), ("weight_only_8bit", "8-bit weight-only (deployed)")):
        p = parity.get(k, {})
        l = get(lat, "aura_sense", k) or {}
        rows.append([label, fmt(p.get("size_mb"), 1), fmt(p.get("embedding_cosine_mean"), 4), fmt(p.get("emotion_top1_agreement"), 3),
                     fmt(p.get("max_abs_risk_prob_diff"), 3), fmt(get(l, "threads_1", "median_ms"), 1), fmt(get(l, "threads_4", "median_ms"), 1)])
    return {"id": "runtime", "title": "Export fidelity and CPU latency of Aura-Sense",
            "caption": f"Agreement with the PyTorch model on 300 held-out sentences, one sentence per call. Latency is the median per sentence "
                       f"({get(lat, 'mean_tokens') or '—'} tokens on average) on {get(lat, 'machine', 'cpu') or 'the test machine'}.",
            "columns": ["Graph", "Size (MB)", "Embedding cos", "Emotion top-1 agree", "Max Δ risk", "1 thread (ms)", "4 threads (ms)"],
            "rows": rows, "highlight": [1]}


LATEX_SUBS = [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"),
              ("±", "$\\pm$"), ("ρ", "$\\rho$"), ("↑", "$\\uparrow$"), ("↓", "$\\downarrow$"), ("≥", "$\\geq$"),
              ("Δ", "$\\Delta$"), ("—", "--"), ("–", "--"), ("·", "$\\cdot$"), ("→", "$\\rightarrow$")]


def tex(s):
    s = str(s)
    for a, b in LATEX_SUBS:
        s = s.replace(a, b)
    return s


def to_latex(t, wide=False):
    env = "table*" if wide else "table"
    align = "l" + "r" * (len(t["columns"]) - 1)
    lines = [f"\\begin{{{env}}}[t]", "\\centering", "\\footnotesize", f"\\caption{{{tex(t['title'])}. {tex(t['caption'])}}}",
             f"\\label{{tab:{t['id']}}}", "\\setlength{\\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(f"\\textbf{{{tex(c)}}}" for c in t["columns"]) + " \\\\", "\\midrule"]
    for i, r in enumerate(t["rows"]):
        cells = [tex(c) for c in r]
        if i in t.get("highlight", []):
            cells = [f"\\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def to_markdown(t):
    out = [f"### {t['title']}", "", f"_{t['caption']}_", "", "| " + " | ".join(t["columns"]) + " |",
           "|" + "|".join(["---"] + ["---:"] * (len(t["columns"]) - 1)) + "|"]
    for i, r in enumerate(t["rows"]):
        cells = [f"**{c}**" if i in t.get("highlight", []) else str(c) for c in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- figures

INK, PEACH, LAV, GREY = "#16140f", "#d9825f", "#8c84c7", "#b9b2a6"


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#6b665e", "axes.labelcolor": INK, "xtick.color": "#4a463f", "ytick.color": "#4a463f",
        "axes.grid": True, "grid.color": "#e6e1d8", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False,
    })
    return plt


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=220)


def fig_ablation(runs, teacher):
    plt = style()
    order = [("mt", "Multi-task"), ("mt_distill", "+ distillation"), ("circ_uw", "+ circumplex\n(uncertainty-wt.)"), ("aura_sense", "Aura-Sense")]
    order = [(k, l) for k, l in order if k in runs]
    f1 = [np.mean([r["emotion"]["macro_f1"] for r in runs[k]]) for k, _ in order]
    sts = [np.mean([r["retrieval"]["stsb_spearman"] for r in runs[k]]) for k, _ in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    x = np.arange(len(order))
    colors = [PEACH if k == "aura_sense" else GREY for k, _ in order]
    axes[0].bar(x, f1, color=colors, width=0.62)
    axes[0].set_title("GoEmotions macro-F1 (28 classes)", fontsize=9)
    axes[1].bar(x, sts, color=colors, width=0.62)
    if teacher:
        axes[1].axhline(teacher, color=LAV, ls="--", lw=1)
        axes[1].text(-0.4, teacher + 0.006, "frozen encoder", color=LAV, fontsize=7.5)
    axes[1].set_title("STS-B Spearman (retrieval geometry)", fontsize=9)
    for ax, vals in zip(axes, (f1, sts)):
        ax.set_xticks(x, [l for _, l in order], fontsize=7.2)
        ax.set_ylim(min(vals) * 0.85, max(vals) * 1.09)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_ablation")
    plt.close(fig)


def fig_circumplex(cal_circ):
    from common import GOEMO_LABELS, load_nrc_vad

    nrc = load_nrc_vad()
    if not nrc or not cal_circ:
        return
    plt = style()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    learned = np.array([cal_circ[l] for l in GOEMO_LABELS])
    human = np.array([nrc[l] for l in GOEMO_LABELS])
    show = {"joy", "love", "gratitude", "excitement", "pride", "relief", "optimism", "amusement", "sadness", "grief", "fear",
            "nervousness", "anger", "disgust", "remorse", "disappointment", "embarrassment", "caring", "neutral", "surprise"}
    for ax, pts, title in ((axes[0], learned, "Learned by Aura-Sense (from EmoBank)"), (axes[1], human, "NRC-VAD human norms (never seen)")):
        ax.axhline(0, color="#cfc8bc", lw=0.8)
        ax.axvline(0, color="#cfc8bc", lw=0.8)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=[PEACH if v >= 0 else LAV for v in pts[:, 0]], zorder=3)
        for (v, a), l in zip(pts, GOEMO_LABELS):
            if l in show:
                ax.annotate(l, (v, a), xytext=(3, 2), textcoords="offset points", fontsize=6.5, color="#3a372f")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("valence")
        ax.set_ylabel("arousal")
        lim = np.abs(pts).max() * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    fig.tight_layout()
    save(fig, "fig_circumplex")
    plt.close(fig)


def fig_agem(agem):
    plt = style()
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    for name, m in agem["methods"].items():
        ours = name == "AGEM (full)"
        ax.scatter(m["isr"], m["recall@3"], s=64 if ours else 34, color=PEACH if ours else GREY, edgecolor=INK if ours else "none", zorder=3)
        ax.annotate(name, (m["isr"], m["recall@3"]), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("inappropriate surfacing of painful memories (ISR, lower is better)")
    ax.set_ylabel("recall@3 on relevant queries")
    ax.set_xlim(-0.03, 1.1)
    ax.set_ylim(-0.03, 1.08)
    fig.tight_layout()
    save(fig, "fig_agem")
    plt.close(fig)


def fig_safety(safety):
    plt = style()
    modes = [("lexicon", GREY), ("classifier", LAV), ("hybrid", PEACH)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    panels = (
        (safety["systems"], "Development probe", [("explicit_crisis_recall", "explicit → crisis"), ("risk_alert_recall", "any risk → alert"), ("benign_false_alert", "benign → alert")]),
        (safety["heldout"]["systems"], "Held-out probe", [("explicit_crisis_recall", "explicit → crisis"), ("risk_alert_recall", "any risk → alert"), ("idiom_false_alert", "idiom → alert")]),
    )
    for ax, (data, title, metrics) in zip(axes, panels):
        x = np.arange(len(metrics))
        for i, (mode, c) in enumerate(modes):
            ax.bar(x + (i - 1) * 0.26, [data[mode][k] for k, _ in metrics], width=0.25, color=c, label=mode)
        ax.set_xticks(x, [l for _, l in metrics], fontsize=7.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.08)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_safety")
    plt.close(fig)


def fig_face(face):
    plt = style()
    cm = np.array(face["aura_face_confusion"], float)
    cm = cm / cm.sum(1, keepdims=True)
    labels = face["labels"]
    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    ax.grid(False)
    im = ax.imshow(cm, cmap="Oranges", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save(fig, "fig_face_confusion")
    plt.close(fig)


# --------------------------------------------------------------------------- website samples


def samples_and_routes():
    from neurocare.affect import fuse
    from neurocare.personas import PERSONAS, route
    from neurocare.safety import SafetyGate
    from neurocare.sense import AuraSense

    sense, gate = AuraSense(), SafetyGate()
    texts = ["I finally finished my project and my team loved it", "I can't stop worrying that I'll fail my exam on Friday",
             "My grandmother passed away last month and I miss her every day", "I'm so angry they cancelled on me again",
             "Everyone at work must think I'm useless", "I'm exhausted, I haven't slept properly in days"]
    samples, routes = [], []
    for t in texts:
        r = sense.read(t)[0]
        samples.append({"text": t, "top": [[k, round(v, 3)] for k, v in r.affect.top(4)]})
        tier = gate.assess(t, r.affect.risk, sense.risk_elevated, sense.risk_crisis).tier
        state = fuse(r.affect, None, None, None, tier)
        pid, why = route(state, t, "empathetic")
        routes.append({"text": t, "dominant": state.dominant, "quadrant": state.quadrant,
                       "persona": pid, "persona_name": PERSONAS[pid].name, "reason": why})
    return samples, routes


def write_numbers(path, runs, face, agem, safety, v1, parity, lat, teacher):
    """Every number quoted in the paper's prose, as LaTeX macros, so text and results cannot drift."""

    def mean(cfg, *p):
        vals = [get(r, *p) for r in runs.get(cfg, []) if get(r, *p) is not None]
        return float(np.mean(vals)) if vals else None

    def sd(cfg, *p):
        vals = [get(r, *p) for r in runs.get(cfg, []) if get(r, *p) is not None]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    m = lambda name: get(agem, "methods", name)  # noqa: E731
    plain = next((k for k in (agem or {}).get("methods", {}) if k.endswith("(plain RAG)")), None)
    report_k = next((k for k in (agem or {}).get("methods", {}) if "0.82" in k), None)
    nums = {
        "Seeds": len(runs.get("aura_sense", [])),
        "EmoF": mean("aura_sense", "emotion", "macro_f1"), "EmoFsd": sd("aura_sense", "emotion", "macro_f1"),
        "Ekman": mean("aura_sense", "emotion", "ekman_macro_f1"),
        "EmoMT": mean("mt", "emotion", "macro_f1"), "EmoDistill": mean("mt_distill", "emotion", "macro_f1"),
        "EmoUW": mean("circ_uw", "emotion", "macro_f1"),
        "UWweight": get(runs.get("circ_uw", [{}])[0], "task_weights", "circumplex"),
        "ValR": mean("aura_sense", "va", "pearson_valence"), "AroR": mean("aura_sense", "va", "pearson_arousal"),
        "ValRcirc": mean("aura_sense", "va", "circumplex_pearson_valence"), "AroRcirc": mean("aura_sense", "va", "circumplex_pearson_arousal"),
        "RiskAUROC": mean("aura_sense", "risk", "auroc"), "DistF": mean("aura_sense", "distortion", "f1"),
        "STS": mean("aura_sense", "retrieval", "stsb_spearman"), "STSmt": mean("mt", "retrieval", "stsb_spearman"),
        "STSteacher": teacher,
        "NRCv": mean("aura_sense", "circumplex_vs_nrc", "pearson_valence"), "NRCa": mean("aura_sense", "circumplex_vs_nrc", "pearson_arousal"),
        "FaceAcc": get(face, "models", "aura_face_ensemble", "accuracy"), "FaceF": get(face, "models", "aura_face_ensemble", "macro_f1"),
        "FaceGB": get(face, "models", "gboost_raw52", "accuracy"), "FaceFACS": get(face, "models", "facs_rules_zero_shot", "accuracy"),
        "FaceDet": (get(face, "face_detection_rate", "test") or 0) * 100,
        "AgemRecall": get(m("AGEM (full)"), "recall@3"), "AgemISR": get(m("AGEM (full)"), "isr"),
        "PlainISR": get(m(plain), "isr") if plain else None, "PlainRecall": get(m(plain), "recall@3") if plain else None,
        "PlainRedund": get(m(plain), "redundancy") if plain else None,
        "NaiveISR": get(m("naive top-1 (original code)"), "isr"), "ReportRecall": get(m(report_k), "recall@3") if report_k else None,
        "SafeDevCrisis": get(safety, "systems", "hybrid", "explicit_crisis_recall"),
        "SafeHeldCrisis": get(safety, "heldout", "systems", "hybrid", "explicit_crisis_recall"),
        "SafeHeldAlert": get(safety, "heldout", "systems", "hybrid", "risk_alert_recall"),
        "SafeVoneHeldCrisis": get(v1, "heldout", "systems", "hybrid", "explicit_crisis_recall"),
        "SafeClsCrisis": get(safety, "systems", "classifier", "explicit_crisis_recall"),
        "SafeClsAlert": get(safety, "systems", "classifier", "risk_alert_recall"),
        "SafeLexAlert": get(safety, "systems", "lexicon", "risk_alert_recall"),
        "SafeHybAlert": get(safety, "systems", "hybrid", "risk_alert_recall"),
        "SafeBenign": get(safety, "systems", "hybrid", "benign_false_alert"),
        "ParityCos": get(parity, "weight_only_8bit", "embedding_cosine_mean"), "ParityTop": get(parity, "weight_only_8bit", "emotion_top1_agreement"),
        "ParityRisk": get(parity, "weight_only_8bit", "max_abs_risk_prob_diff"),
        "SizeQ": get(parity, "weight_only_8bit", "size_mb"), "SizeF": get(parity, "fp32", "size_mb"),
        "LatQ": get(lat, "aura_sense", "weight_only_8bit", "threads_4", "median_ms"), "LatF": get(lat, "aura_sense", "fp32", "threads_4", "median_ms"),
        "AgemThousand": get(lat, "agem", "1000", "median_ms"), "AgemTenK": get(lat, "agem", "10000", "median_ms"),
    }
    lines = ["% generated by research/make_summary.py -- do not edit by hand"]
    for k, v in nums.items():
        if v is None:
            val = "--"
        elif isinstance(v, int):
            val = str(v)
        elif k in ("FaceDet", "SizeQ", "SizeF", "UWweight") or k.startswith("Lat") or k.startswith("Agem") and k.endswith(("Thousand", "TenK")):
            val = f"{v:.1f}"
        elif k == "ParityCos":
            val = f"{v:.4f}"
        elif k.startswith(("Safe", "Agem", "Plain", "Naive", "Report")):
            val = f"{v:.2f}"
        else:
            val = f"{v:.3f}"
        lines.append(f"\\newcommand{{\\{k}}}{{{val}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    runs = aura_runs()
    heads, ft = load("heads_report.json"), load("finetune_report.json")
    face, agem = load("aura_face_report.json"), load("agem_report.json")
    safety, v1 = load("safety_report.json"), load("safety_report_lexicon_v1.json")
    parity, lat = load("aura_sense_onnx_parity.json") or {}, load("latency_report.json")
    teacher = teacher_stsb()
    final = runs.get("aura_sense", [])
    main_run = next((r for r in final if r["seed"] == 13), final[0] if final else None)

    tables = [text_table(runs, heads, ft, teacher)]
    if face:
        tables.append(face_table(face))
    if agem:
        tables.append(agem_table(agem))
    if safety and "heldout" in safety:
        tables.append(safety_table(safety, v1))
    if parity:
        tables.append(runtime_table(parity, lat))
    (RESULTS / "tables.md").write_text("\n".join(to_markdown(t) for t in tables), encoding="utf-8")
    paper = ROOT.parent / "paper"
    (paper / "tables").mkdir(parents=True, exist_ok=True)
    for t in tables:
        (paper / "tables" / f"{t['id']}.tex").write_text(to_latex(t, wide=t["id"] in ("text", "safety", "runtime")), encoding="utf-8")

    if runs:
        fig_ablation(runs, teacher)
    if main_run:
        fig_circumplex(main_run["calibration"]["circumplex"])
    if agem:
        fig_agem(agem)
    if safety and "heldout" in safety:
        fig_safety(safety)
    if face:
        fig_face(face)

    write_numbers(paper / "numbers.tex", runs, face, agem, safety, v1, parity, lat, teacher)

    import shutil

    (paper / "figures").mkdir(exist_ok=True)
    for f in FIGURES.glob("*.pdf"):
        shutil.copy(f, paper / "figures" / f.name)

    samples, routes = samples_and_routes()
    m = main_run or {}
    reduction = get(agem, "isr_reduction_vs_plain_rag")
    summary = {
        "aura_sense": {
            "emotion_macro_f1": get(m, "emotion", "macro_f1"), "ekman_macro_f1": get(m, "emotion", "ekman_macro_f1"),
            "valence_r": get(m, "va", "pearson_valence"), "arousal_r": get(m, "va", "pearson_arousal"),
            "valence_r_circ": get(m, "va", "circumplex_pearson_valence"), "arousal_r_circ": get(m, "va", "circumplex_pearson_arousal"),
            "risk_auroc": get(m, "risk", "auroc"), "risk_n": 10000, "distortion_f1": get(m, "distortion", "f1"),
            "stsb": get(m, "retrieval", "stsb_spearman"), "stsb_teacher": round(teacher, 4) if teacher else None,
            "circ_nrc_valence_r": get(m, "circumplex_vs_nrc", "pearson_valence"),
            "circ_nrc_arousal_r": get(m, "circumplex_vs_nrc", "pearson_arousal"),
            "size_mb": round(get(parity, "weight_only_8bit", "size_mb")) if get(parity, "weight_only_8bit", "size_mb") else None,
        },
        "aura_face": {"accuracy": get(face, "models", "aura_face_ensemble", "accuracy"),
                      "macro_f1": get(face, "models", "aura_face_ensemble", "macro_f1"), "chance": 0.125} if face else {},
        "agem": {"demo": agem["demo"], "isr_reduction": f"{reduction * 100:.0f}%" if reduction else None,
                 "isr_plain": get(agem, "methods", next((k for k in agem["methods"] if k.endswith("(plain RAG)")), ""), "isr"),
                 "isr_agem": get(agem, "methods", "AGEM (full)", "isr")} if agem else {},
        "samples": samples, "routes": routes, "tables": tables,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print((RESULTS / "tables.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
