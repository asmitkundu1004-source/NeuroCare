### Text affect: baselines, ablations and Aura-Sense

_Held-out test sets: GoEmotions (5,427), EmoBank (1,000), Reddit risk (10,000), Shreevastava–Foltz (506), STS-B (1,379). For circumplex models, valence and arousal are read through the learned circumplex._

| Model | Encoder passes / turn | Emotion macro-F1 | Ekman macro-F1 | Valence r | Arousal r | Risk AUROC | Distortion F1 | STS-B ρ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic / ridge (per task) | — | 0.443 | 0.564 | 0.509 | 0.271 | 0.980 | 0.826 | — |
| Frozen MiniLM + task heads | 1 | 0.396 | 0.520 | 0.625 | 0.436 | 0.985 | 0.832 | 0.820 |
| MiniLM fine-tuned separately per task | one per task | 0.435 | 0.600 | — | — | 0.996 | — | — |
| Multi-task fine-tune | 1 | 0.519 | 0.637 | 0.731 | 0.503 | 0.995 | 0.826 | 0.643 |
| + retrieval-preserving distillation (n=3 seeds) | 1 | 0.490 ± 0.002 | 0.612 ± 0.006 | 0.699 ± 0.005 | 0.484 ± 0.009 | 0.995 ± 0.000 | 0.834 ± 0.004 | 0.831 ± 0.001 |
| + circumplex, uncertainty-weighted | 1 | 0.396 | 0.548 | 0.730 | 0.511 | 0.995 | 0.837 | 0.828 |
| **Aura-Sense (ours): + circumplex, fixed weight (n=3 seeds)** | **1** | **0.492 ± 0.002** | **0.617 ± 0.008** | **0.689 ± 0.002** | **0.382 ± 0.002** | **0.995 ± 0.000** | **0.836 ± 0.004** | **0.830 ± 0.001** |

### Facial affect from blendshapes only (AffectNet, 8 classes)

_Test n = 5,788, chance accuracy 0.125. Faces detected in 99.6% of images. No model sees pixels._

| Model | Accuracy | Balanced acc. | Macro-F1 |
|---|---:|---:|---:|
| FACS prototype rules (zero-shot) | 0.375 | 0.316 | 0.279 |
| Logistic regression, 52 coefficients | 0.462 | 0.449 | 0.443 |
| Gradient boosting, 52 coefficients | 0.512 | 0.498 | 0.495 |
| MLP, 52 coefficients | 0.506 | 0.488 | 0.487 |
| MLP + bilateral encoding | 0.507 | 0.483 | 0.487 |
| MLP + mirror augmentation | 0.508 | 0.490 | 0.489 |
| MLP + both | 0.507 | 0.483 | 0.487 |
| **Aura-Face: 5-member ensemble (deployed)** | **0.516** | **0.492** | **0.494** |

### Memory policies on a scripted personal history

_20 memories (10 painful), 40 relevant, 15 unrelated and 10 tangential queries, 6 follow-up turns. ISR: share of unrelated or tangential queries where a painful memory was injected (lower is better)._

| Policy | Recall@3 ↑ | ISR ↓ | Tangential pain ↓ | Silent when unrelated ↑ | Redundancy ↓ |
|---|---:|---:|---:|---:|---:|
| naive top-1 (original code) | 1.00 | 0.36 | 0.50 | 0.00 | 0.67 |
| top-3, cos >= 0.82 (progress report) | 0.07 | 0.00 | 0.00 | 1.00 | 0.00 |
| top-3, cos >= 0.38 (plain RAG) | 0.95 | 0.28 | 0.60 | 0.60 | 0.67 |
| AGEM without affect gate | 0.95 | 0.28 | 0.60 | 0.60 | 0.00 |
| AGEM without refractory | 0.95 | 0.00 | 0.00 | 0.60 | 0.50 |
| **AGEM (full)** | **0.95** | **0.00** | **0.00** | **0.60** | **0.00** |

### Safety gate on short conversational messages

_Development probe: 20 explicit, 12 indirect, 12 distress, 25 benign. Held-out probe written before lexicon v2: 10 explicit, 8 indirect, 10 idioms. No configuration escalated ordinary distress to a crisis._

| System | Dev crisis recall | Dev alert recall | Dev benign false alert | Held-out crisis recall | Held-out alert recall | Held-out idiom false alert |
|---|---:|---:|---:|---:|---:|---:|
| lexicon · lexicon v1 | 0.90 | 0.78 | 0.00 | 0.50 | 0.39 | 0.00 |
| classifier · lexicon v1 | 0.00 | 0.56 | 0.04 | 0.00 | 0.56 | 0.00 |
| hybrid · lexicon v1 | 0.90 | 0.88 | 0.04 | 0.50 | 0.72 | 0.00 |
| lexicon · lexicon v2 | 1.00 | 0.94 | 0.00 | 0.90 | 0.94 | 0.00 |
| classifier · lexicon v2 | 0.00 | 0.56 | 0.04 | 0.00 | 0.56 | 0.00 |
| **hybrid · lexicon v2** | **1.00** | **1.00** | **0.04** | **0.90** | **1.00** | **0.00** |

### Export fidelity and CPU latency of Aura-Sense

_Agreement with the PyTorch model on 300 held-out sentences, one sentence per call. Latency is the median per sentence (19.8 tokens on average) on Intel Core i5-11400H (6 cores), 16 GB RAM._

| Graph | Size (MB) | Embedding cos | Emotion top-1 agree | Max Δ risk | 1 thread (ms) | 4 threads (ms) |
|---|---:|---:|---:|---:|---:|---:|
| fp32 | 90.5 | 1.0000 | 1.000 | 0.000 | 8.1 | 4.1 |
| **8-bit weight-only (deployed)** | **24.2** | **0.9996** | **0.990** | **0.084** | **47.2** | **27.8** |
