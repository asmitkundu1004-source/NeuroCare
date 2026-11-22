# Running NeuroCare on your own computer

There are two ways to run it. Start with **A**; it takes five minutes and needs no AI setup.
Do **B** when you want the fine-tuned Aura language model and the neural voice.

---

## What to install first

| Tool | Needed for | Download |
|---|---|---|
| **Node.js 20 or newer** (LTS) | the website (A and B) | https://nodejs.org |
| **Python 3.11** | the server (B only) | https://www.python.org/downloads/ (tick "Add python.exe to PATH" in the installer) |
| **Ollama** | the fine-tuned Aura model (B only) | https://ollama.com/download |
| **Chrome or Edge** | voice and camera features | |

Check they installed: open PowerShell and run `node -v`, `python --version`, `ollama -v`.

Unzip the project somewhere simple, for example `C:\NeuroCare`. The commands below assume that folder.

---

## A. Website only (everything runs in the browser)

```powershell
cd C:\NeuroCare\web
npm install          # first time only, takes 1–2 minutes
npm run dev
```

Open **http://localhost:5173** in Chrome or Edge and click **Begin**.

Works in this mode: emotion reading, safety and helplines, memory, Insights, Breathe, Model Lab, Privacy, camera face sensing, voice input (hold the mic button or the Space key) and spoken replies (browser voice).
Replies come from a short built-in responder. For real AI replies in this mode: in the session, click the status pill at the top-left and load **Llama 3.2 1B** (about 0.9 GB, downloaded once, needs Chrome or Edge on a laptop).

Stop the website with `Ctrl + C` in the terminal.

---

## B. Full version with the fine-tuned Aura model

### One-time setup

**1. Put the model file in place.** The fine-tuned model is a separate 1.9 GB file
(`aura-empathy-llama-3.2-3b.Q4_K_M.gguf`, shared separately via Google Drive or a pen drive).
Copy it into:

```
C:\NeuroCare\models\aura-llm\
```

The file name must stay exactly `aura-empathy-llama-3.2-3b.Q4_K_M.gguf`.

**2. Register it with Ollama** (make sure the Ollama app is running; look for its icon in the taskbar tray):

```powershell
cd C:\NeuroCare\models\aura-llm
ollama create aura -f Modelfile
ollama list            # should show "aura"
```

**3. Set up the server:**

```powershell
cd C:\NeuroCare\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pykokoro   # optional: natural neural voice
```

If `activate` fails with "running scripts is disabled", run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**4. Set up the website:**

```powershell
cd C:\NeuroCare\web
npm install
```

### Every time you run it

Open **two** PowerShell windows.

Window 1, the server:
```powershell
cd C:\NeuroCare\backend
.venv\Scripts\activate
uvicorn neurocare.server:app --port 8000
```
Wait for `Aura-Sense loaded ... ready`.

Window 2, the website:
```powershell
cd C:\NeuroCare\web
npm run dev
```

Open **http://localhost:5173** and click **Begin**. The status pill at the top-left should read
**"Local server · aura · fine-tuned Llama 3.2 3B"**. Replies now come from the fine-tuned model.

**One-window alternative:** run `npm run build` once in `web`, then only start the server and open **http://localhost:8000**.

---

## Quick test

1. Type "I'm anxious about my exam on Friday". A reply appears; on a laptop the right-hand panel shows the emotion reading, the chosen stance and why.
2. Hold the **mic** button (or Space), speak, release. Your words appear and Aura speaks back.
3. Turn on the **camera** icon. A small round preview shows your expression, read on the device.
4. Open **Model Lab** and **Insights** from the home page menu.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| Status pill says "On this device" in version B | The server is not running, or not on port 8000. Check window 1. |
| Status pill says "Reflective responder (no LLM running)" | Ollama is not running, or `ollama create aura -f Modelfile` was skipped. |
| `ollama create` says file not found | The `.gguf` is not in `models\aura-llm\`, or its name is different. |
| No voice input | Use Chrome or Edge and allow the microphone. Firefox does not support it. |
| No spoken reply | Check the speaker icon at the top of the session is not muted, and the system volume. |
| Port already in use | Close the old terminal, or use `--port 8001` and open that port. |
| `npm` or `python` not recognised | Reinstall Node.js / Python and restart PowerShell. |

More detail: `docs\DEPLOYMENT.md` (hosting) and `docs\DEVELOPMENT.md` (retraining the models and reproducing every result).
