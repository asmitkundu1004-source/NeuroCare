# Deploying NeuroCare

NeuroCare runs in two shapes, and you can host either for free.

| Shape | What runs where | Best for | Free hosts |
|---|---|---|---|
| **Zero-server site** | Everything in the visitor's browser: Aura-Sense (WebAssembly), Aura-Face, safety gate, AGEM memory in an encrypted IndexedDB vault, optional Llama 3.2 on WebGPU | A public demo link, the viva, a portfolio | Vercel, Netlify, GitHub Pages, Cloudflare Pages |
| **Full stack** | The FastAPI server with the fine-tuned Aura model (Ollama), serving the same site from one port | Showing the fine-tuned LLM and neural voice | Your laptop, Hugging Face Spaces (Docker), Render |

The site looks for a server automatically and switches to the in-browser engine when there is none, so a static deployment never breaks.

---

## 1. Put the code on GitHub

1. Create an empty repository on GitHub (no README, no .gitignore).
2. Open the `NeuroCare-Aura` folder, select everything inside it (including the hidden `.github` folder and `.gitignore`) and drag it onto the repository page ("uploading an existing file").
   - If dot-files are hidden, show them first. On Windows: View → Show → Hidden items.
   - GitHub's browser upload accepts files up to 25 MB and 100 files per upload. The largest file, `backend/models/aura-sense/model.onnx`, is 24.2 MB. If the page complains about the file count, upload in two batches (for example `web/` first, then the rest).
3. Commit to `main`.

From the command line instead, inside `NeuroCare-Aura`:

```bash
git init && git add . && git commit -m "NeuroCare Aura 2.0"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

---

## 2. Zero-server site

### Vercel (recommended)

1. Go to [vercel.com/new](https://vercel.com/new) and import the repository.
2. Set **Root Directory** to `web`. Everything else comes from `web/vercel.json`.
3. Deploy. You get `https://<project>.vercel.app` in about a minute.

### Netlify

1. [app.netlify.com](https://app.netlify.com) → Add new site → Import an existing project → choose the repository.
2. Leave every field empty: `netlify.toml` at the repository root sets the base directory, build command and output folder.
3. Deploy.

### GitHub Pages

1. In the repository: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push to `main`, or run the workflow from the Actions tab. `.github/workflows/pages.yml` runs the browser-parity tests, builds with the right base path and publishes to `https://<you>.github.io/<repo>/`.

### Cloudflare Pages

Framework preset *Vite*, root directory `web`, build command `npm run build`, output directory `dist`. Add a file `web/public/_redirects` containing `/* /index.html 200` so deep links work.

### What visitors get

- Sensing, safety, memory, Insights, Breathe, the Model Lab and the privacy centre work immediately.
- Replies come from the reflective responder until the visitor loads a language model in the tab: Session → status pill → *Language model in this tab*. That needs Chrome or Edge with WebGPU, downloads about 0.9 GB (Llama 3.2 1B) once, and is cached afterwards.
- Camera and microphone need HTTPS, which every host above provides.

---

## 3. Full stack on your own computer

The best experience: the fine-tuned Aura model, Kokoro voice and an encrypted database file on your own disk.

**Requirements:** Python 3.10+, Node.js 20+, [Ollama](https://ollama.com/download), about 4 GB of free RAM while chatting.

```bash
# 1. the language model: copy the fine-tuned GGUF into models/aura-llm/
#    as aura-empathy-llama-3.2-3b.Q4_K_M.gguf, then
cd models/aura-llm
ollama create aura -f Modelfile

# 2. the server
cd ../../backend
python -m venv .venv
.venv\Scripts\activate              # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install pykokoro                 # optional neural voice
uvicorn neurocare.server:app --port 8000

# 3. the site, in a second terminal
cd web
npm install
npm run dev                          # http://localhost:5173
```

The Vite dev server forwards `/api` and `/ws` to port 8000, so the site finds the server by itself. To serve everything from the server instead, run `npm run build` once; the server then serves `web/dist` at `http://localhost:8000`.

### Configuration

All optional, set as environment variables before starting the server:

| Variable | Default | Meaning |
|---|---|---|
| `NEUROCARE_LLM_URL` | `http://localhost:11434` | Ollama (or OpenAI-compatible) endpoint |
| `NEUROCARE_LLM_MODEL` | `aura` | Model name |
| `NEUROCARE_LLM_API` | `ollama` | `ollama`, or `openai` for llama.cpp server, LM Studio, vLLM |
| `NEUROCARE_PASSPHRASE` | unset | Derive the database key from a passphrase (scrypt) instead of a random key file |
| `NEUROCARE_DATA_DIR` | `backend/data` | Where `neurocare.db` and its key live |
| `NEUROCARE_REGION` | `IN` | Default helpline region: `IN`, `US` or `UK` |
| `NEUROCARE_TTS` | `1` | `0` turns Kokoro off |
| `NEUROCARE_CORS` | `http://localhost:5173` | Comma-separated allowed origins |

---

## 4. Full stack on Hugging Face Spaces (free, public)

A free CPU Space has 2 vCPU and 16 GB RAM: enough for the 3B model at a few tokens per second, which is fine for a demo.

1. **Upload the model once.** Create a model repository on Hugging Face (for example `<you>/aura-empathy-3b`), upload the `.gguf` through the web page and copy its download link:
   `https://huggingface.co/<you>/aura-empathy-3b/resolve/main/aura-empathy-llama-3.2-3b.Q4_K_M.gguf`
2. **Create a Space** → SDK *Docker* → blank template.
3. Upload the contents of this repository to the Space (the same drag-and-drop as GitHub). Then edit the Space's `README.md` so that it **begins** with:
   ```yaml
   ---
   title: NeuroCare
   colorFrom: indigo
   colorTo: pink
   sdk: docker
   app_port: 7860
   ---
   ```
4. Space **Settings → Variables and secrets → New variable**: `AURA_GGUF_URL` = the link from step 1.
5. The Space builds the root `Dockerfile`, downloads the model on first start and serves the site and API at `https://<you>-<space>.hf.space`.

Free Spaces sleep when idle and lose their disk on restart. For a demo that is fine; the vault starts empty again.

### Render (alternative)

New → Web Service → connect the repository → Runtime *Docker*. The free instance has 512 MB of RAM, too little for the 3B model, so leave `AURA_GGUF_URL` unset there; the server runs with the reflective responder.

---

## 5. Pointing the static site at a remote server

To make a Vercel or Netlify site use a Space instead of the in-browser engine, set this on the static host and redeploy:

```
VITE_API_URL=https://<you>-<space>.hf.space
```

The Docker image already allows any origin (`NEUROCARE_CORS=*`). Narrow it to your site's URL if you prefer.

---

## Before sharing a link

- [ ] Open `/session`, send a message, check the status pill.
- [ ] Open `/privacy` and confirm "requests to other servers" stays at 0 during a conversation.
- [ ] Type a crisis phrase and confirm the helplines for your region appear.
- [ ] Try it on a phone over HTTPS: hold-to-talk, camera toggle, bottom sheets.
