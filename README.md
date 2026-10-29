# 🚗 KreyatiSky TrafficVision

> Système intelligent, ultra-léger et robuste de détection et de comptage directionnel de véhicules en temps réel avec **YOLO**, **FastAPI** et **Dashboard interactif**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8_Nano-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌟 Fonctionnalités Clés

- 🎯 **Comptage Directionnel Précis** :
  - ⬆️ **Véhicules qui montent** (Sens Nord / In)
  - ⬇️ **Véhicules qui descendent** (Sens Sud / Out)
  - 🚘 **Total des passages** & Débit en temps réel (véhicules/minute)
- 👁️ **IA & Vision par Ordinateur Ultra-Légère** :
  - Détection basée sur **YOLO11 Nano** (`yolo11n.pt`), optimisé pour CPU, GPU et Jetson (TensorRT).
  - Suivi multi-objets persistant (**ByteTrack**) avec identifiant unique par véhicule pour éliminer tout double comptage.
  - Détection et classification automatique : **Voitures, Camions, Motos, Bus**.
- 📹 **Compatibilité Multi-Sources Vidéo** :
  - **Webcam de l'ordinateur** (caméra intégrée ou USB externe).
  - **Caméras Réseau / Flux IP** (RTSP, HTTP, MJPEG) avec reconnexion automatique en cas de coupure.
  - **Fichiers Vidéo Locaux** (`.mp4`, `.avi`, `.mov`) avec téléversement direct depuis le tableau de bord.
  - **Simulateur de trafic intégré (Mode Démo)** : permet d'utiliser et tester l'application immédiatement sans caméra branchée.
- 💻 **Dashboard Web Moderne & Réactif** :
  - Flux vidéo en direct à faible latence (MJPEG).
  - Cartes KPI dynamiques, pourcentages et jauges de répartition.
  - Journal d'activité des passages en temps réel.
  - Réglage à chaud : déplacement de la ligne de comptage avec curseur en direct, réglage du seuil de confiance, filtrage des véhicules.
  - Actions rapides : capture d'instantanés (Snapshots), export des données au format CSV, réinitialisation des compteurs.

---

## 🚀 Démarrage Rapide en 1 Clic

### Option A : Lancement Automatique (Recommandé)

Ouvrez votre terminal dans le dossier du projet et lancez :

```bash
./run.sh
```

Ce script va automatiquement :
1. Créer l'environnement virtuel Python (`.venv`).
2. Installer les dépendances (`requirements.txt`).
3. Lancer le serveur et ouvrir votre navigateur sur `http://localhost:8000`.

---

### Option B : Installation Manuelle

1. **Créer et activer un environnement virtuel** :
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

3. **Lancer l'application** :
   ```bash
   python3 app.py
   ```

4. **Accéder au Dashboard** :
   Ouvrez votre navigateur web à l'adresse : **[http://localhost:8000](http://localhost:8000)**

---

## ⚙️ Configuration des Sources Vidéo

Dans le tableau de bord, cliquez sur **Paramètres** pour sélectionner votre source :

### 1. Webcam Locale
- Sélectionnez `Webcam`
- Indiquez `0` (pour la webcam intégrée) ou `1`, `2` (pour une caméra USB externe).

### 2. Caméra IP / Flux Réseau RTSP
- Sélectionnez `Caméra IP / Flux Réseau (RTSP)`
- Entrez l'URL de votre flux, par exemple :
  ```text
  rtsp://admin:motdepasse@192.168.1.50:554/stream1
  ```
  ou un flux HTTP MJPEG : `http://192.168.1.60:8080/video`

### 3. Fichier Vidéo Local
- Sélectionnez `Fichier Vidéo Local`
- Choisissez un fichier `.mp4` / `.avi` / `.mov` sur votre ordinateur et cliquez sur **Changer la Source**.

### 4. Mode Démo (Simulateur)
- Sélectionnez `Simulateur de trafic (Démo)` pour tester le système avec une route simulée en temps réel.

### Comptage en direct avec une caméra : matériel conseillé

Le suivi des véhicules demande **au moins 5 images analysées par seconde** ; 10 ou plus
est idéal. La cadence réelle s'affiche sur le badge FPS du dashboard (« max » = capacité
mesurée de l'ordinateur, survol = matériel utilisé). Sous 5 i/s, le mode automatique
réduit la largeur d'analyse pour garder un suivi fiable.

- **Mac M1 / M2 / M3 / M4** : la puce graphique est utilisée automatiquement.
- **PC avec carte NVIDIA** : utilisée automatiquement si PyTorch est installé avec CUDA.
  `pip install -r requirements.txt` installe souvent la version processeur seule : installer
  alors la version CUDA indiquée sur https://pytorch.org avant de lancer `app.py`.
  Le terminal affiche au démarrage `calcul : cuda:0` (NVIDIA), `mps` (Mac) ou `cpu`.
- **Windows** : `run.sh` ne fonctionne pas ; suivre l'« Option B », avec
  `.venv\Scripts\activate` à la place de `source .venv/bin/activate`.
- Une caméra qui se coupe est reconnectée automatiquement ; l'analyse traite toujours
  l'image la plus récente, sans prendre de retard sur le direct.
- Ordinateur puissant : le modèle `yolo11s.pt` (plus précis, ~3× plus lent) peut remplacer
  `yolo11n.pt` dans `config.json` (champ `model_name`), puis redémarrer.

### NVIDIA Jetson Orin Nano (JetPack 6)

Le Jetson est idéal pour une caméra fixe au bord de la route : faible consommation,
analyse sur sa carte graphique avec TensorRT.

1. Installer JetPack 6 (Ubuntu 22.04, Python 3.10) sur le Jetson, copier le projet, puis :
   ```bash
   ./setup_jetson.sh            # installation complète (~20 min)
   ./setup_jetson.sh --service  # idem + démarrage automatique à l'allumage
   ```
   Le script installe la version NVIDIA de PyTorch (celle de `pip` n'utilise pas la carte
   graphique), garde l'OpenCV de JetPack (GStreamer, pour la caméra CSI) et compile le modèle
   en **TensorRT FP16** (`yolo11n.engine`), activé automatiquement dans `config.json`.
2. Performances maximales : `sudo nvpmodel -m 0 && sudo jetson_clocks`
3. Lancer : `./run.sh`, puis ouvrir `http://<adresse-du-jetson>:8000` depuis un autre appareil.

Sources de caméra sur le Jetson :
- **Caméra CSI** (sur nappe, ex. Raspberry Pi Camera v2 / IMX219) : source « Caméra CSI du
  Jetson », index `0` (ou `1` pour le second connecteur).
- **Caméra USB** : source « Webcam », index `0`.
- **Caméra IP** : source « Caméra IP / Flux Réseau », URL RTSP ou HTTP.

Remarques :
- Un moteur `.engine` est propre au Jetson qui l'a compilé : ne pas le copier sur un autre
  appareil. Pour revenir au modèle PyTorch : `"model_name": "yolo11n.pt"` dans `config.json`.
- Avec un moteur TensorRT, la largeur d'analyse est fixée à la taille d'export (640 px).
- Service : `sudo systemctl status kreyatisky`, journal : `journalctl -u kreyatisky -f`.

---

## 📁 Structure du Projet

```
KreyatiSky/
├── app.py                  # Serveur FastAPI, flux vidéo MJPEG et API REST
├── core/
│   ├── __init__.py
│   ├── config.py           # Gestionnaire de configuration dynamique
│   ├── detector.py         # Moteur de détection YOLOv8 et rendu visuel HUD
│   ├── counter.py          # Logique de franchissement de ligne & direction (Up/Down)
│   └── video_stream.py     # Lecteur multi-sources (Webcam, RTSP, Fichiers, Démo)
├── static/
│   ├── css/
│   │   └── style.css       # Thème sombre Cyber-Glassmorphism
│   └── js/
│       └── dashboard.js    # Contrôleur JavaScript interactif
├── templates/
│   └── index.html          # Interface utilisateur du Dashboard
├── requirements.txt        # Dépendances Python
├── run.sh                  # Script de démarrage en 1 clic
├── setup_jetson.sh         # Installation sur NVIDIA Jetson (PyTorch GPU, TensorRT, caméra CSI)
├── test_counter.py         # Tests unitaires et géométriques
└── README.md               # Documentation
```

---

## 📊 Endpoints de l'API REST

| Méthode | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Page principale du Dashboard web |
| `GET` | `/video_feed` | Flux vidéo MJPEG avec détections en temps réel |
| `GET` | `/api/stats` | Données JSON des compteurs, FPS et historique |
| `GET` | `/api/settings` | Paramètres actuels du système |
| `POST` | `/api/settings` | Mise à jour des réglages (seuil, ligne, classes) |
| `POST` | `/api/source` | Changement de la source vidéo |
| `POST` | `/api/upload_video` | Téléversement d'un fichier vidéo |
| `POST` | `/api/reset` | Réinitialisation des compteurs |
| `GET` | `/api/snapshot` | Capture d'un instantané JPEG |
| `GET` | `/api/export/csv` | Téléchargement du rapport au format CSV |

## Déploiement serveur

- Lancez un seul worker Uvicorn par caméra (`uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1`).
  Le modèle, le tracker et le compteur résident en mémoire et ne doivent pas être dupliqués.
- Placez l'application derrière un reverse proxy avec authentification et HTTPS, surtout si les URL RTSP
  contiennent des identifiants.
- Une webcam sélectionnée sur le serveur désigne la webcam physiquement connectée au serveur, pas celle du
  navigateur. Pour une caméra distante, utilisez RTSP/HTTP.
- `auto_performance=true` adapte automatiquement la cadence d'analyse à la puissance disponible.
  `target_fps` sert uniquement de limite lorsque le mode automatique est désactivé. `process_width=640`
  et `jpeg_quality=75` constituent le profil équilibré recommandé.
- Les fichiers importés sont analysés image par image; les flux réseau/webcam utilisent la trame la plus
  récente afin d'éviter l'accumulation de retard.

### Calibration de la vitesse

Dans **Paramètres → Calibration de la vitesse**, placez la ligne bleue sur un second repère de la route et
indiquez la distance réelle en mètres entre cette ligne et la ligne jaune. La vitesse est calculée avec le
temps du média (`distance / durée × 3,6`) et affichée en km/h par véhicule. La précision dépend directement
de la mesure réelle saisie et de la stabilité de l'identifiant de suivi entre les deux lignes.
