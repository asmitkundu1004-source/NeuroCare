# Development and reproducing the research

Everything reported on the website, in the paper and in the project report is produced by the scripts in `research/`. This page lists them in order, with the time each took on the development machine (Intel Core i5-11400H, 16 GB RAM, NVIDIA RTX 3050 Laptop GPU with 4 GB).

## Layout

```
NeuroCare-Aura/
├── backend/                 FastAPI server and the Python runtime of every model
│   ├── neurocare/
│   │   ├── sense.py         Aura-Sense ONNX runtime (embedding + affect in one pass)
│   │   ├── affect.py        multimodal late fusion on the valence–arousal plane
│   │   ├── safety.py        three-tier safety gate (lexicon + risk head)
│   │   ├── agem.py          Affect-Gated Episodic Memory ranking (pure numpy)
│   │   ├── memory.py        AES-256-GCM encrypted SQLite store and mood log
│   │   ├── personas.py      six therapeutic stances and the affect router
│   │   ├── llm.py           streaming client for Ollama / OpenAI-compatible servers
│   │   ├── tts.py           optional Kokoro-82M voice, sentence by sentence
│   │   └── server.py        REST + WebSocket server; also serves the built site
│   └── models/              Aura-Sense (ONNX, 24 MB) and the shared safety lexicon
├── web/                     React site; src/engine/ is the browser port of the backend
├── research/                training, evaluation and figures
├── models/aura-llm/         Ollama Modelfile for the fine-tuned Llama 3.2 3B
├── paper/                   IEEE-style paper (LaTeX)
└── docs/                    deployment, development, project report
```

## Environment

```bash
cd research
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu126      # or the CPU wheel
pip install sentence-transformers datasets scikit-learn pandas matplotlib scipy ^
            onnx onnxruntime onnx_ir tokenizers mediapipe==0.10.14 opencv-python-headless cryptography
```

Datasets download from Hugging Face on first use, except two small files placed in `research/data/`:

| File | Source | Licence |
|---|---|---|
| `emobank.csv` | [JULIELab/EmoBank](https://github.com/JULIELab/EmoBank), `corpus/emobank.csv` | CC BY-SA 4.0 |
| `nrc/` | [NRC VAD Lexicon](https://saifmohammad.com/WebPages/nrc-vad.html), unzipped | research use |

## Pipeline

| Step | Command | Output | Time |
|---|---|---|---|
| Frozen-encoder baselines (TF-IDF, MiniLM + heads) | `python train_heads.py` | `results/heads_report.json` | ~15 min |
| Per-task fine-tuning upper bound | `python finetune.py emotion risk` | `results/finetune_report.json` | ~20 min |
| **Aura-Sense**, final model and ONNX export | `python train_aura_sense.py --config aura_sense --export` | `backend/models/aura-sense/` | ~16 min |
| Ablations | `python train_aura_sense.py --config mt,mt_distill,circ_uw` | `results/aura_sense_*.json` | ~45 min |
| Extra seeds | `python train_aura_sense.py --config aura_sense,mt_distill --seeds 7,29` | same | ~60 min |
| Re-export without retraining | `python train_aura_sense.py --config aura_sense --reexport` | ONNX + parity report | ~2 min |
| Blendshapes from AffectNet images | `python face/extract_blendshapes.py` | `cache/affectnet_blendshapes_*.npz` | ~10 min |
| **Aura-Face** | `python face/train_aura_face.py` | `web/public/models/aura-face.json` | ~8 min |
| Safety probe | `python eval_safety.py` | `results/safety_report.json` | 1 min |
| AGEM benchmark | `python eval_memory.py` | `results/agem_report.json` | 1 min |
| Latency | `python bench_latency.py` | `results/latency_report.json` | 2 min |
| Browser parity fixture | `python make_web_fixture.py` | `web/tests/fixtures/sense_fixture.json` | seconds |
| Tables, figures, website summary | `python make_summary.py` | `results/tables.md`, `figures/`, `results/summary.json` | 1 min |

`make_summary.py` is the single source of every number shown anywhere. After running it, rebuild the site (`npm run build` in `web/`) so the landing page and the Model Lab pick up the new `summary.json`.

Seeds are fixed (13, plus 7 and 29 for the variance runs). Results reproduce to within GPU non-determinism.

## Tests

```bash
cd web && npm run sync && npm test
```

This checks that the browser runtime reproduces the Python server: identical WordPiece token ids, and model outputs within 1e-3. Run `python research/make_web_fixture.py` after re-exporting Aura-Sense.

## Running the backend in development

```bash
cd backend
uvicorn neurocare.server:app --reload --port 8000
```

The backend needs only `backend/requirements.txt`. Without Ollama it still senses, gates, remembers and answers with the reflective responder.

## Editing the safety lexicon

`backend/models/safety_lexicon.json` is shared by the server and the browser (`npm run sync` copies it). Patterns must work in both Python `re` and JavaScript `RegExp`, so no look-behind. Write backslashes as they should appear in JSON (`\\b`). After editing, run `python research/eval_safety.py` and compare with the stored report.
