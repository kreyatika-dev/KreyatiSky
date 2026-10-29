#!/bin/bash
# Installation de KreyatiSky TrafficVision sur NVIDIA Jetson Orin Nano (JetPack 6.x).
#
# Ce que ce script fait, et pourquoi :
#  - L'environnement Python voit les paquets système (--system-site-packages) : TensorRT
#    et l'OpenCV de JetPack (compilé avec GStreamer, indispensable à la caméra CSI) ne
#    s'installent pas avec pip.
#  - Le PyTorch installé par pip sur ARM n'utilise pas la carte graphique : il est
#    remplacé par la version NVIDIA pour Jetson (versions du guide officiel Ultralytics,
#    https://docs.ultralytics.com/guides/nvidia-jetson/ , section JetPack 6.1).
#  - Le modèle YOLO11n est compilé en moteur TensorRT FP16 (yolo11n.engine), 3 à 4 fois
#    plus rapide que PyTorch sur le Jetson. Un moteur n'est valable que sur l'appareil
#    qui l'a compilé : ne pas le copier d'un Jetson à un autre.
#
# Usage : ./setup_jetson.sh            (installation complète)
#         ./setup_jetson.sh --service  (installe aussi le démarrage automatique au boot)

set -euo pipefail
cd "$(dirname "$0")"

TORCH_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl"
TORCHVISION_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl"
ONNXRUNTIME_WHL="https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl"
CUDSS_DEB="cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb"
CUDSS_URL="https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/$CUDSS_DEB"
MODEL="yolo11n"

if [ ! -f /etc/nv_tegra_release ]; then
    echo "❌ Ce script est prévu pour un NVIDIA Jetson (JetPack 6). Sur un autre ordinateur : ./run.sh"
    exit 1
fi
echo "Jetson détecté : $(head -n 1 /etc/nv_tegra_release)"

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if [ "$PYV" != "3.10" ]; then
    echo "❌ Python 3.10 attendu (JetPack 6, Ubuntu 22.04) ; trouvé Python $PYV."
    echo "   Les versions NVIDIA de PyTorch utilisées ici n'existent que pour Python 3.10."
    exit 1
fi

echo "== 1/6 Paquets système =="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv python3-opencv libopenblas-dev

echo "== 2/6 Environnement Python =="
# Un .venv créé par run.sh ne voit pas les paquets système : on le recrée.
if [ -d .venv ] && ! grep -q "include-system-site-packages = true" .venv/pyvenv.cfg 2>/dev/null; then
    echo "Environnement existant sans accès aux paquets JetPack : recréation."
    rm -rf .venv
fi
if [ ! -d .venv ]; then
    python3 -m venv --system-site-packages .venv
fi
source .venv/bin/activate
pip install -U pip

echo "== 3/6 PyTorch NVIDIA pour Jetson =="
# Installé avant le reste : ultralytics le trouve déjà présent et n'en télécharge pas
# une autre version (celle de pip n'utiliserait pas la carte graphique).
pip install "$TORCH_WHL" "$TORCHVISION_WHL" "$ONNXRUNTIME_WHL"
# Bibliothèque requise par torch 2.10 sur Jetson (guide Ultralytics)
if ! dpkg -s cudss >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    wget -q -O "$tmp/$CUDSS_DEB" "$CUDSS_URL"
    sudo dpkg -i "$tmp/$CUDSS_DEB"
    sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cudss
    rm -rf "$tmp"
fi
# Les roues NVIDIA sont compilées pour NumPy 1.x
pip install "numpy<2.0.0,>=1.23.0"

echo "== 4/6 Dépendances du projet =="
pip install -r requirements.txt
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA indisponible'; print('PyTorch', torch.__version__, '- GPU :', torch.cuda.get_device_name(0))"

echo "== 5/6 OpenCV avec GStreamer (caméra CSI) =="
# pip a installé un OpenCV sans GStreamer qui masquerait celui de JetPack.
HAS_GST='import cv2, sys; sys.exit(0 if any("GStreamer" in l and "YES" in l for l in cv2.getBuildInformation().splitlines()) else 1)'
if /usr/bin/python3 -c "$HAS_GST" 2>/dev/null; then
    pip uninstall -y opencv-python opencv-python-headless || true
    echo "OpenCV de JetPack utilisé (GStreamer disponible : caméra CSI possible)."
else
    echo "⚠️  OpenCV de JetPack sans GStreamer : caméras USB et IP utilisables, pas la caméra CSI."
fi
python3 -c "import cv2; print('OpenCV', cv2.__version__)"

echo "== 6/6 Modèle TensorRT (quelques minutes) =="
if python3 -c "import tensorrt" 2>/dev/null; then
    if [ ! -f "$MODEL.engine" ]; then
        python3 -c "from ultralytics import YOLO; YOLO('$MODEL.pt').export(format='engine', quantize=16, imgsz=640, device=0)"
    fi
    python3 -c "from core.config import ConfigManager; ConfigManager().update({'model_name': '$MODEL.engine'})"
    echo "Modèle actif : $MODEL.engine (TensorRT FP16)"
else
    echo "⚠️  TensorRT introuvable en Python : le modèle reste $MODEL.pt (carte graphique via PyTorch)."
    echo "   Pour l'installer : sudo apt-get install -y nvidia-tensorrt, puis relancer ce script."
fi

if [ "${1:-}" = "--service" ]; then
    echo "== Démarrage automatique =="
    sudo tee /etc/systemd/system/kreyatisky.service >/dev/null <<EOF
[Unit]
Description=KreyatiSky TrafficVision
After=network-online.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$PWD
ExecStart=$PWD/.venv/bin/python3 $PWD/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    sudo systemctl daemon-reload
    sudo systemctl enable --now kreyatisky.service
    echo "Service actif : sudo systemctl status kreyatisky  |  journal : journalctl -u kreyatisky -f"
fi

echo
echo "✅ Installation terminée. Lancer : ./run.sh"
echo "   Performances maximales (recommandé pour une caméra en direct) :"
echo "   sudo nvpmodel -m 0 && sudo jetson_clocks"
