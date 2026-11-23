# Deploying NeuroCare

NeuroCare runs in two shapes, and you can host either for free.

| Shape | What runs where | Best for | Free hosts |
|---|---|---|---|
| **Zero-server site** | Everything in the visitor's browser: Aura-Sense (WebAssembly), Aura-Face, safety gate, AGEM memory in an encrypted IndexedDB vault, optional Llama 3.2 on WebGPU | A public demo link, the viva, a portfolio | Vercel, Netlify, GitHub Pages, Cloudflare Pages |
| **Full stack** | The FastAPI server with the fine-tuned Aura model (Ollama), serving the same site from one port | Showing the fine-tuned LLM and neural voice | Kaggle (free GPU), Oracle Cloud Always Free, your laptop + Cloudflare Tunnel |

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

## 4. Full stack on Kaggle (free GPU, no card)

Kaggle notebooks come with a free NVIDIA T4 GPU (about 30 hours a week, up to 12 hours per run) and need no payment card.
`deploy/kaggle/neurocare_kaggle.ipynb` runs the whole stack there (the fine-tuned Aura model on the GPU, the server and the site)
and prints a public `https://….trycloudflare.com` link. Replies take a second or two.

**One-time setup**

1. Sign in at https://www.kaggle.com and verify your phone number (**Settings → Phone verification**). Without it, notebooks have no internet.
2. **Upload the model:** **Datasets → New Dataset**, drop in `aura-empathy-llama-3.2-3b.Q4_K_M.gguf` (1.9 GB), name it `aura-model`, keep it **Private**, **Create**.
3. **Import the notebook:** **Code → New Notebook**, then **File → Import Notebook** and upload `deploy/kaggle/neurocare_kaggle.ipynb`.
4. In the notebook's right-hand panel:
   - **Add Input → Your Work → Datasets → aura-model**
   - **Session options → Accelerator: GPU T4 x2**
   - **Session options → Internet: On**

**Each time you need the link**

- **With your laptop off:** **Save Version → Save & Run All (Commit) → Save**. Open the running version from the notebook's
  **Versions / Logs** view; after about 5 minutes the log shows `NeuroCare is live: https://….trycloudflare.com`.
  It stays up for about 11.5 hours, then stops by itself.
- **Quick test:** **Run All** and keep the tab open; the link is printed under the last cell.
- **Link on your phone:** set `NTFY_TOPIC` in the second cell to a long random word, install the free *ntfy* app
  (or open `https://ntfy.sh/<word>`), and the link is pushed there when it is ready.

The link changes on every run, so start it shortly before a demo. Memories live only for that run.
Stop a run early from **View Active Events → Stop session** to save GPU hours.

---

## 5. Full stack on Oracle Cloud (free, always on, needs a card)

Hugging Face now charges for Docker Spaces, so the free home for the full stack is Oracle Cloud's
**Always Free** tier: an Ampere server with 4 CPU cores and 24 GB of RAM that never sleeps.
It runs the fine-tuned 3B model on the CPU at about 5 to 8 seconds per reply. Sign-up asks for a card
to verify identity; Always Free resources are not charged.

**1. Create the server.**
- Sign up at https://www.oracle.com/cloud/free/ and choose an Indian home region (Hyderabad or Mumbai).
- **Compute → Instances → Create instance.** Image: *Canonical Ubuntu 22.04*. Shape: *Ampere VM.Standard.A1.Flex*, 4 OCPU, 24 GB.
  Under *Add SSH keys* choose *Generate a key pair* and **download the private key** (save it as `oracle.key`).
  If Oracle says "Out of capacity", try another availability domain or try again later.
- Open the instance → its **subnet** → **Security List** → **Add Ingress Rules**: source `0.0.0.0/0`, TCP, destination port `80`. Add a second rule for `443`.
- Copy the instance's **public IP**.

**2. Get a free web address.** Microphone and camera need HTTPS, which needs a name.
Log in at https://www.duckdns.org, create a sub-domain (for example `neurocare-aura`) and set its IP to the public IP.

**3. Copy the code and the model to the server.** From PowerShell on your computer:

```powershell
ssh -i oracle.key ubuntu@PUBLIC_IP "git clone https://github.com/asmitkundu1004-source/NeuroCare.git"
scp -i oracle.key models\aura-llm\aura-empathy-llama-3.2-3b.Q4_K_M.gguf ubuntu@PUBLIC_IP:NeuroCare/models/aura-llm/
```

(Run the `scp` line from the project folder, or give the full path to the `.gguf`. It is 1.9 GB.)
If `ssh` refuses the key as "too open", run `icacls oracle.key /inheritance:r /grant:r "$($env:USERNAME):R"` once.

**4. Run the setup script** on the server. It installs Ollama, Node, Python packages and Caddy,
registers the model, builds the site, starts everything as a service and gets the HTTPS certificate:

```powershell
ssh -i oracle.key ubuntu@PUBLIC_IP
bash NeuroCare/deploy/oracle-setup.sh neurocare-aura.duckdns.org
```

It takes about 10 minutes and ends with `Open https://neurocare-aura.duckdns.org`. The session's status pill should read
**"Local server · aura · fine-tuned Llama 3.2 3B"**. The server restarts by itself after a reboot.

**Updating** after new commits: `cd NeuroCare && git pull && bash deploy/oracle-setup.sh neurocare-aura.duckdns.org`

**Logs:** `journalctl -u neurocare -n 50` (server), `journalctl -u caddy -n 50` (HTTPS).

The server keeps a single encrypted memory store, so everyone using one server link shares it.
Before a demo, clear it from the Privacy page.

### Other options

- **Your laptop + Cloudflare Tunnel** (no account): start the server as in section 3 after `npm run build`, then
  `cloudflared tunnel --url http://localhost:8000` prints a public `https://….trycloudflare.com` link. The laptop must stay awake and the link changes each time.
- **Docker** (any paid host, or a Hugging Face PRO Space): the root `Dockerfile` serves everything on port 7860; set `AURA_GGUF_URL` to a download link for the `.gguf`.
- **Render free tier**: 512 MB of RAM is too little for the 3B model; leave `AURA_GGUF_URL` unset and the server runs with the reflective responder.

---

## 6. Pointing the static site at a remote server

To make the Vercel or Netlify site use your server instead of the in-browser engine, set this on the static host and redeploy:

```
VITE_API_URL=https://neurocare-aura.duckdns.org
```

The setup script and the Docker image already allow any origin (`NEUROCARE_CORS=*`). Narrow it to your site's URL if you prefer.

---

## Before sharing a link

- [ ] Open `/session`, send a message, check the status pill.
- [ ] Open `/privacy` and confirm "requests to other servers" stays at 0 during a conversation.
- [ ] Type a crisis phrase and confirm the helplines for your region appear.
- [ ] Try it on a phone over HTTPS: hold-to-talk, camera toggle, bottom sheets.
