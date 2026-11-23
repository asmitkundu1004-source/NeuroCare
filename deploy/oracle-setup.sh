#!/usr/bin/env bash
# One-time setup of the full NeuroCare stack (fine-tuned Aura model, server, site, HTTPS)
# on a fresh Ubuntu 22.04/24.04 server, such as an Oracle Cloud Always Free Ampere VM.
#
#   bash deploy/oracle-setup.sh your-name.duckdns.org
#
# Run it again at any time to update after a `git pull`; every step is safe to repeat.
set -euo pipefail

DOMAIN="${1:?usage: bash deploy/oracle-setup.sh your-name.duckdns.org}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGUF="$ROOT/models/aura-llm/aura-empathy-llama-3.2-3b.Q4_K_M.gguf"
export DEBIAN_FRONTEND=noninteractive

if [ ! -f "$GGUF" ]; then
  echo "Model file missing. Copy it to:"
  echo "  $GGUF"
  echo "(see docs/DEPLOYMENT.md, section 4), then run this script again."
  exit 1
fi

step() { printf '\n== %s\n' "$1"; }

step "System packages"
sudo apt-get update -y
sudo -E apt-get install -y git curl gpg python3-venv python3-pip iptables-persistent

if ! node -v 2>/dev/null | grep -Eq '^v(2[0-9]|[3-9][0-9])\.'; then
  step "Node.js 20"
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo -E apt-get install -y nodejs
fi

if ! command -v caddy >/dev/null; then
  step "Caddy (HTTPS)"
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y
  sudo -E apt-get install -y caddy
fi

if ! command -v ollama >/dev/null; then
  step "Ollama"
  curl -fsSL https://ollama.com/install.sh | sh
fi

step "Fine-tuned Aura model"
i=0
until curl -sf http://127.0.0.1:11434/api/tags >/dev/null; do
  i=$((i + 1)); [ "$i" -gt 60 ] && { echo "Ollama did not start"; exit 1; }
  sleep 1
done
(cd "$ROOT/models/aura-llm" && ollama create aura -f Modelfile)

step "Server"
python3 -m venv "$ROOT/backend/.venv"
"$ROOT/backend/.venv/bin/pip" install --upgrade pip -q
"$ROOT/backend/.venv/bin/pip" install -r "$ROOT/backend/requirements.txt" -q

step "Website"
(cd "$ROOT/web" && npm ci --no-audit --no-fund && npm run build)

step "Service (starts on boot)"
sudo tee /etc/systemd/system/neurocare.service >/dev/null <<EOF
[Unit]
Description=NeuroCare server
After=network-online.target ollama.service
Wants=ollama.service

[Service]
User=$(id -un)
WorkingDirectory=$ROOT/backend
Environment=NEUROCARE_CORS=*
ExecStart=$ROOT/backend/.venv/bin/uvicorn neurocare.server:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable neurocare >/dev/null
sudo systemctl restart neurocare

step "Firewall (ports 80 and 443)"
for port in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 1 -p tcp --dport "$port" -j ACCEPT
done
sudo netfilter-persistent save >/dev/null

step "HTTPS for $DOMAIN"
printf '%s {\n\treverse_proxy 127.0.0.1:8000\n}\n' "$DOMAIN" | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl restart caddy

step "Checking"
i=0
until curl -sf http://127.0.0.1:8000/api/health >/dev/null; do
  i=$((i + 1)); [ "$i" -gt 90 ] && { echo "Server did not answer. Logs: journalctl -u neurocare -n 50"; exit 1; }
  sleep 1
done
echo "Server is up."
echo
echo "Open https://$DOMAIN"
echo "(The first visit can take a minute while the HTTPS certificate is issued.)"
