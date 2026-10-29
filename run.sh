#!/bin/bash

set -e

cd "$(dirname "$0")"

echo "======================================================"
echo "🚗 Lancement de KreyatiSky TrafficVision..."
echo "======================================================"

if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ Erreur: python3 n'est pas installe."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "📦 Creation de l'environnement virtuel (.venv)..."
    python3 -m venv .venv
fi

source .venv/bin/activate

# Jetson : PyTorch, TensorRT et OpenCV viennent de JetPack. Un « pip install » ordinaire
# les remplacerait par des versions qui n'utilisent pas la carte graphique.
if [ -f /etc/nv_tegra_release ]; then
    if ! python3 -c "import torch, fastapi, ultralytics, anthropic; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        echo "❌ Jetson détecté mais pas encore installé (ou PyTorch sans GPU)."
        echo "   Lancez d'abord : ./setup_jetson.sh"
        exit 1
    fi
    echo "⚡ Jetson prêt (carte graphique NVIDIA)."
# Verification rapide : si tout est deja installe, demarrage instantane sans passer par pip !
elif python3 -c "import fastapi, uvicorn, ultralytics, cv2, numpy, anthropic; assert numpy.__version__.startswith('1.')" >/dev/null 2>&1; then
    echo "⚡ Tous les modules sont prets ! Demarrage instantane..."
else
    echo "🔄 Installation des modules..."
    pip install -r requirements.txt
fi

LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)

echo "======================================================"
echo "✅ Prêt ! Lancement du serveur TrafficVision..."
echo "💻 Sur cet ordinateur   : http://localhost:8000"
if [ -n "$LOCAL_IP" ]; then
    echo "📱 Sur votre réseau Wi-Fi/LAN : http://$LOCAL_IP:8000"
fi
echo "======================================================"

python3 app.py

