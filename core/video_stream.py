import cv2
import time
import threading
import numpy as np
import random
import os
from typing import Optional, Tuple

# Configuration du timeout RTSP (3 secondes max si la caméra IP est hors-ligne)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;3000000"

LIVE_SOURCES = ("rtsp", "http", "webcam", "csi")
# Une caméra IP met souvent plusieurs secondes à négocier le flux (RTSP, première
# image clé), une webcam USB à s'allumer : un délai trop court ferait échouer
# chaque tentative de connexion, indéfiniment.
OPEN_TIMEOUT_WEBCAM_S = 6.0
OPEN_TIMEOUT_NETWORK_S = 10.0
# Une image perdue isolée ne justifie pas de couper la caméra.
LIVE_READ_FAILURES = 5
# Résolution demandée à une webcam (OpenCV ouvre sinon en 640x480).
WEBCAM_SIZE = (1280, 720)
CSI_SIZE, CSI_FPS = (1920, 1080), 30


def csi_pipeline(sensor_id: int) -> str:
    """Pipeline GStreamer d'une caméra CSI de Jetson (caméra sur nappe, ex. IMX219).

    nvarguscamerasrc passe par le processeur d'image du Jetson ; appsink ne garde
    que l'image la plus récente pour ne pas prendre de retard sur le direct.
    """
    width, height = CSI_SIZE
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, framerate={CSI_FPS}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def opencv_has_gstreamer() -> bool:
    for line in cv2.getBuildInformation().splitlines():
        if "GStreamer" in line:
            return "YES" in line
    return False



class DemoTrafficGenerator:
    """
    Générateur de flux vidéo synthétique de trafic routier.
    Permet de tester l'application immédiatement sans caméra ni fichier externe.
    Simule une route à double sens avec véhicules montants et descendants.
    """

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.vehicles = []
        self.frame_idx = 0
        self.next_vehicle_id = 1
        self.colors = [
            (220, 50, 50),   # Rouge
            (50, 180, 50),   # Vert
            (50, 100, 220),  # Bleu
            (220, 220, 220), # Blanc/Gris
            (30, 30, 30),    # Noir
            (240, 200, 50)   # Jaune
        ]
        self._init_vehicles()

    def _init_vehicles(self):
        # Création initiale de quelques véhicules avec IDs uniques
        self.vehicles = [
            {"id": 1, "lane": "up", "y": 400, "speed": 3.5, "w": 40, "h": 70, "color": (50, 100, 220), "type": "car"},
            {"id": 2, "lane": "down", "y": 50, "speed": 4.0, "w": 42, "h": 75, "color": (220, 50, 50), "type": "car"},
            {"id": 3, "lane": "up", "y": 250, "speed": 3.0, "w": 48, "h": 110, "color": (220, 220, 220), "type": "truck"},
            {"id": 4, "lane": "down", "y": 180, "speed": 4.5, "w": 25, "h": 45, "color": (50, 200, 50), "type": "motorcycle"}
        ]
        self.next_vehicle_id = 5

    def get_frame(self) -> np.ndarray:
        self.frame_idx += 1
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 1. Décor : Herbe / Paysage
        img[:] = (40, 60, 40)

        # 2. Chaussée (Route goudronnée au milieu)
        road_left = int(self.width * 0.25)
        road_right = int(self.width * 0.75)
        img[:, road_left:road_right] = (60, 60, 65)

        # Bordures blanches de la route
        cv2.line(img, (road_left, 0), (road_left, self.height), (230, 230, 230), 4)
        cv2.line(img, (road_right, 0), (road_right, self.height), (230, 230, 230), 4)

        # Ligne médiane pointillée
        road_mid = (road_left + road_right) // 2
        dash_len = 25
        gap_len = 20
        offset = (self.frame_idx * 3) % (dash_len + gap_len)
        for y in range(-offset, self.height, dash_len + gap_len):
            cv2.line(img, (road_mid, max(0, y)), (road_mid, min(self.height, y + dash_len)), (240, 220, 0), 2)

        # 3. Voies de circulation
        lane_up_center = (road_left + road_mid) // 2
        lane_down_center = (road_mid + road_right) // 2

        # 4. Génération aléatoire de nouveaux véhicules
        if random.random() < 0.025 and len(self.vehicles) < 7:
            lane = random.choice(["up", "down"])
            v_type = random.choices(["car", "truck", "motorcycle", "bus"], weights=[65, 15, 10, 10])[0]
            
            if v_type == "car":
                vw, vh = 42, random.randint(70, 85)
                spd = random.uniform(3.0, 4.5)
            elif v_type == "truck":
                vw, vh = 48, random.randint(110, 140)
                spd = random.uniform(2.2, 3.2)
            elif v_type == "bus":
                vw, vh = 48, random.randint(120, 150)
                spd = random.uniform(2.5, 3.5)
            else: # motorcycle
                vw, vh = 24, random.randint(45, 55)
                spd = random.uniform(4.0, 5.5)

            new_v = {
                "id": self.next_vehicle_id,
                "lane": lane,
                "y": self.height + 60 if lane == "up" else -150,
                "speed": spd,
                "w": vw,
                "h": vh,
                "color": random.choice(self.colors),
                "type": v_type
            }
            self.next_vehicle_id += 1
            self.vehicles.append(new_v)

        # 5. Déplacement et rendu des véhicules
        remaining_vehicles = []
        for v in self.vehicles:
            if v["lane"] == "up":
                v["y"] -= v["speed"]
                cx = lane_up_center
            else:
                v["y"] += v["speed"]
                cx = lane_down_center

            x1 = int(cx - v["w"] // 2)
            y1 = int(v["y"] - v["h"] // 2)
            x2 = int(cx + v["w"] // 2)
            y2 = int(v["y"] + v["h"] // 2)

            # Dessin du véhicule avec carrosserie, pare-brise et phares
            if -180 < v["y"] < self.height + 180:
                remaining_vehicles.append(v)
                
                # Carrosserie
                cv2.rectangle(img, (x1, y1), (x2, y2), v["color"], -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)

                # Pare-brise avant & arrière
                windshield_color = (180, 200, 220)
                margin_x = max(3, v["w"] // 8)
                ws_h = max(5, v["h"] // 6)
                
                if v["lane"] == "up":
                    # Avant en haut
                    cv2.rectangle(img, (x1 + margin_x, y1 + ws_h // 2), (x2 - margin_x, y1 + ws_h * 2), windshield_color, -1)
                    # Phares avant
                    cv2.circle(img, (x1 + 6, y1 + 3), 3, (200, 255, 255), -1)
                    cv2.circle(img, (x2 - 6, y1 + 3), 3, (200, 255, 255), -1)
                    # Feux arrière (rouges)
                    cv2.rectangle(img, (x1 + 3, y2 - 4), (x1 + 10, y2), (0, 0, 240), -1)
                    cv2.rectangle(img, (x2 - 10, y2 - 4), (x2 - 3, y2), (0, 0, 240), -1)
                else:
                    # Avant en bas
                    cv2.rectangle(img, (x1 + margin_x, y2 - ws_h * 2), (x2 - margin_x, y2 - ws_h // 2), windshield_color, -1)
                    # Phares avant
                    cv2.circle(img, (x1 + 6, y2 - 3), 3, (200, 255, 255), -1)
                    cv2.circle(img, (x2 - 6, y2 - 3), 3, (200, 255, 255), -1)
                    # Feux arrière
                    cv2.rectangle(img, (x1 + 3, y1), (x1 + 10, y1 + 4), (0, 0, 240), -1)
                    cv2.rectangle(img, (x2 - 10, y1), (x2 - 3, y1 + 4), (0, 0, 240), -1)

                # Toit
                roof_h = max(10, v["h"] // 3)
                roof_y1 = y1 + (v["h"] - roof_h) // 2
                roof_y2 = roof_y1 + roof_h
                cv2.rectangle(img, (x1 + margin_x + 2, roof_y1), (x2 - margin_x - 2, roof_y2), (40, 40, 40), -1)

        self.vehicles = remaining_vehicles
        return img

    def get_active_detections(self) -> list:
        """Retourne les détections des véhicules actifs pour le mode démo."""
        road_left = int(self.width * 0.25)
        road_right = int(self.width * 0.75)
        road_mid = (road_left + road_right) // 2
        lane_up_center = (road_left + road_mid) // 2
        lane_down_center = (road_mid + road_right) // 2

        dets = []
        for v in self.vehicles:
            if 0 <= v["y"] <= self.height:
                cx = lane_up_center if v["lane"] == "up" else lane_down_center
                x1 = int(cx - v["w"] // 2)
                y1 = int(v["y"] - v["h"] // 2)
                x2 = int(cx + v["w"] // 2)
                y2 = int(v["y"] + v["h"] // 2)
                
                dets.append({
                    "id": v["id"],
                    "bbox": [x1, y1, x2, y2],
                    "class": v["type"],
                    "conf": 0.94
                })
        return dets


class VideoStreamManager:
    """
    Gestionnaire universel de flux vidéo avec capture asynchrone multi-threadée.
    Prend en charge Webcam, Fichier Vidéo (lecture du début à la fin avec statistiques finales),
    Flux IP/RTSP et Simulateur Démo.
    """

    def __init__(self, source_type: str = "demo", source_path: str = "0", autostart: bool = True):
        self.source_type = source_type
        self.source_path = source_path
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.demo_gen = DemoTrafficGenerator()
        
        self.current_frame: Optional[np.ndarray] = None
        self.is_running: bool = False
        self.lock = threading.RLock()
        self.capture_lock = threading.Lock()
        # Sérialise les ouvertures de source (changement utilisateur vs reconnexion auto)
        self.source_lock = threading.Lock()
        self.source_generation: int = 0
        self._source_changed = threading.Event()
        self.thread: Optional[threading.Thread] = None
        
        self.width: int = 640
        self.height: int = 480
        self.fps_source: float = 30.0
        self.status_message: str = "Initialisation..."
        self.is_connected: bool = False
        self.frame_sequence: int = 0
        self.processed_sequence: int = 0
        self.reconnect_attempts: int = 0
        self.last_frame_time: Optional[float] = None
        self.last_error: Optional[str] = None
        self._read_failures: int = 0

        # Gestion spécifique aux fichiers vidéo
        self.total_frames: int = 0
        self.current_frame_idx: int = 0
        self.is_paused: bool = False
        self.is_finished: bool = False
        self.auto_loop: bool = False

        if autostart:
            self.start()

    def start(self):
        """Démarre le thread de capture vidéo."""
        if self.is_running:
            return

        self.is_running = True
        self._init_source()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _open_capture_with_timeout(self, source, timeout_sec: float = 2.0,
                                   api: int = cv2.CAP_ANY) -> Optional[cv2.VideoCapture]:
        """Ouvre un flux VideoCapture avec un timeout strict sans bloquer."""
        res = [None]
        expired = threading.Event()
        handoff = threading.Lock()
        def _open():
            try:
                cap = cv2.VideoCapture(source, api)
                if cap and cap.isOpened():
                    with handoff:
                        if expired.is_set():
                            cap.release()
                        else:
                            res[0] = cap
                else:
                    if cap:
                        cap.release()
                    res[0] = None
            except Exception:
                res[0] = None

        th = threading.Thread(target=_open, daemon=True)
        th.start()
        th.join(timeout=timeout_sec)
        with handoff:
            expired.set()
            return res[0]

    def _init_source(self):
        """Initialise la source selon le type configuré."""

        with self.capture_lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

        with self.lock:
            self.is_connected = False
            self.last_error = None
            self.is_finished = False
            self.is_paused = False
            self.current_frame_idx = 0
            self.total_frames = 0
            self.fps_source = 30.0
            self.frame_sequence = 0
            self.processed_sequence = 0

        source_label = os.path.basename(self.source_path) if self.source_type == "file" else self.source_type
        print(f"[VideoStream] Initialisation de la source: type={self.source_type}, source={source_label}")


        if self.source_type == "demo":
            self.status_message = "Simulateur de trafic actif (Démo)"
            self.is_connected = True
            return

        try:
            if self.source_type == "webcam":
                cam_id = int(self.source_path) if str(self.source_path).isdigit() else 0
                self.cap = self._open_capture_with_timeout(cam_id, timeout_sec=OPEN_TIMEOUT_WEBCAM_S)
                if self.cap is not None:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_SIZE[0])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_SIZE[1])
                self.status_message = f"Webcam (Index {cam_id})"
            elif self.source_type == "csi":
                cam_id = int(self.source_path) if str(self.source_path).isdigit() else 0
                if opencv_has_gstreamer():
                    self.cap = self._open_capture_with_timeout(
                        csi_pipeline(cam_id), timeout_sec=OPEN_TIMEOUT_WEBCAM_S, api=cv2.CAP_GSTREAMER)
                    self.status_message = f"Caméra CSI (capteur {cam_id})"
                else:
                    self.cap = None
                    self.last_error = ("OpenCV sans GStreamer : la caméra CSI est inaccessible. "
                                       "Installer avec setup_jetson.sh.")
                    self.status_message = self.last_error
            elif self.source_type == "file":
                if os.path.exists(self.source_path):
                    self.cap = cv2.VideoCapture(self.source_path)
                    self.status_message = f"Fichier vidéo: {os.path.basename(self.source_path)}"
                else:
                    print(f"[VideoStream] Fichier introuvable: {self.source_path}")
                    self.status_message = f"Fichier introuvable: {os.path.basename(self.source_path)}"
                    self.cap = None
            elif self.source_type in ("rtsp", "http"):
                self.cap = self._open_capture_with_timeout(self.source_path, timeout_sec=OPEN_TIMEOUT_NETWORK_S)
                self.status_message = "Flux réseau en cours de connexion"

            
            if self.cap and self.cap.isOpened():
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.width = w
                    self.height = h
                
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                if fps and 0 < fps <= 240:
                    self.fps_source = fps
                if self.source_type == "file":
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.status_message = f"Vidéo chargée : {self.total_frames} trames"

                # Lire immédiatement la première trame pour initialiser l'affichage
                ret, first_frame = self.cap.read()
                if ret and first_frame is not None:
                    with self.lock:
                        self.current_frame = first_frame.copy()
                        self.frame_sequence += 1
                        self.width = first_frame.shape[1]
                        self.height = first_frame.shape[0]
                        self.last_frame_time = time.monotonic()
                        self.is_connected = True
                        self.reconnect_attempts = 0
                    if self.source_type == "file":
                        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    elif self.source_type == "webcam":
                        self.status_message = f"Webcam connectée (Index {cam_id})"
                    elif self.source_type == "csi":
                        self.status_message = f"Caméra CSI connectée (capteur {cam_id})"
                    else:
                        self.status_message = "Flux réseau connecté"
                    print(f"[VideoStream] Source connectée: {self.width}x{self.height}, total_frames={self.total_frames}")
                else:
                    self.last_error = "La source est ouverte mais ne fournit aucune image"
                    self.status_message = self.last_error
                    self.is_connected = False
                    with self.capture_lock:
                        self.cap.release()
                        self.cap = None
            else:
                if not self.last_error:
                    self.last_error = f"Impossible d'ouvrir la source {self.source_type}. Nouvelle tentative en cours."
                self.status_message = self.last_error
                print(f"[VideoStream] Échec d'ouverture de la source {self.source_type}.")
                self.is_connected = False
        except Exception as e:
            self.status_message = f"Erreur source: {type(e).__name__}"
            self.last_error = str(e)
            print(f"[VideoStream] Exception: {e}")
            self.is_connected = False


    def _capture_loop(self):
        """Boucle de capture de trame en arrière-plan."""
        while self.is_running:
            t0 = time.time()
            frame_interval = 1.0 / max(1.0, self.fps_source)

            # Si en pause, on conserve la trame actuelle
            if self.is_paused:
                time.sleep(0.05)
                continue

            if self.source_type == "demo":
                frame = self.demo_gen.get_frame()
                with self.lock:
                    self.current_frame = frame
                    self.frame_sequence += 1
                    self.width = frame.shape[1]
                    self.height = frame.shape[0]
                    self.last_frame_time = time.monotonic()
            else:
                if self.cap is not None and self.cap.isOpened():
                    if self.source_type == "file" and self.is_finished:
                        # La vidéo est terminée : on reste sur la dernière trame
                        time.sleep(0.1)
                        continue

                    # Un fichier est analysé image par image sans perte. Les flux live
                    # gardent au contraire la trame la plus récente pour une faible latence.
                    if self.source_type == "file":
                        with self.lock:
                            waiting_for_processor = self.frame_sequence > self.processed_sequence
                        if waiting_for_processor:
                            time.sleep(0.002)
                            continue

                    with self.capture_lock:
                        ret, frame = self.cap.read() if self.cap is not None else (False, None)
                    if ret and frame is not None:
                        with self.lock:
                            self.current_frame = frame
                            self.frame_sequence += 1
                            self.width = frame.shape[1]
                            self.height = frame.shape[0]
                            self.is_connected = True
                            self.last_frame_time = time.monotonic()
                            self.last_error = None
                            self.reconnect_attempts = 0
                            self._read_failures = 0
                            if self.source_type == "file":
                                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    else:
                        # Fin de fichier vidéo atteinte
                        if self.source_type == "file":
                            if self.auto_loop:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                self.current_frame_idx = 0
                            else:
                                self.is_finished = True
                                self.current_frame_idx = self.total_frames
                                self.status_message = "✅ Analyse terminée (Fin de la vidéo)"
                                print(f"[VideoStream] Fin de la vidéo atteinte ({self.total_frames} frames analysées).")
                        elif self.source_type in LIVE_SOURCES:
                            self._read_failures += 1
                            if self._read_failures >= LIVE_READ_FAILURES:
                                # Tentative de reconnexion réseau
                                self._read_failures = 0
                                self.is_connected = False
                                self._reconnect_after_delay()
                            else:
                                time.sleep(0.05)
                else:
                    if self.source_type == "csi" and not opencv_has_gstreamer():
                        time.sleep(1.0)   # aucune reconnexion ne peut réussir
                    elif self.source_type in LIVE_SOURCES:
                        self._reconnect_after_delay()
                    else:
                        time.sleep(0.25)

            if self.source_type in LIVE_SOURCES:
                # cap.read() attend déjà la prochaine image de la caméra. Ralentir la
                # lecture laisserait les images s'accumuler dans le tampon d'OpenCV
                # et l'analyse prendrait un retard croissant sur le direct.
                continue
            elapsed = time.time() - t0
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)

    def _reconnect_after_delay(self):
        """Rouvre la source après un délai croissant, sauf si l'utilisateur l'a changée entre-temps."""
        generation = self.source_generation
        self.reconnect_attempts += 1
        # Attente interrompue par change_source : la nouvelle source démarre sans délai.
        self._source_changed.wait(min(10.0, 1.0 + self.reconnect_attempts))
        self._source_changed.clear()
        with self.source_lock:
            if generation == self.source_generation:
                self._init_source()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Retourne la dernière trame capturée de façon thread-safe."""
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """Image, numéro et instant de l'image, lus ensemble.

        En direct, l'instant est celui de la capture et non celui de l'analyse :
        la mesure de vitesse ne dépend pas du temps de calcul de chaque image.
        """
        with self.lock:
            if self.source_type == "file" and self.fps_source > 0:
                timeline = self.current_frame_idx / self.fps_source
            else:
                timeline = self.last_frame_time or time.monotonic()
            if self.current_frame is None:
                return False, None, self.frame_sequence, timeline
            return True, self.current_frame.copy(), self.frame_sequence, timeline

    def read_with_sequence(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Retourne une copie et son numéro, afin de ne traiter chaque trame qu'une fois."""
        with self.lock:
            if self.current_frame is None:
                return False, None, self.frame_sequence
            return True, self.current_frame.copy(), self.frame_sequence

    def acknowledge(self, sequence: int):
        with self.lock:
            self.processed_sequence = max(self.processed_sequence, sequence)

    def get_timeline(self) -> float:
        """Temps logique du flux, utilisé pour les statistiques de débit."""
        with self.lock:
            if self.source_type == "file" and self.fps_source > 0:
                return self.current_frame_idx / self.fps_source
        return time.monotonic()

    def change_source(self, source_type: str, source_path: str):
        """Change dynamiquement la source vidéo."""
        with self.source_lock:
            with self.lock:
                self.source_type = source_type
                self.source_path = source_path
                self.current_frame = None
                self.frame_sequence += 1
                self.processed_sequence = self.frame_sequence
                self.is_paused = False
                self.reconnect_attempts = 0
                self.source_generation += 1
            self._init_source()
        self._source_changed.set()

    def replay_video(self):
        """Recommence la lecture de la vidéo depuis le début."""
        with self.lock:
            if self.cap is not None and self.source_type == "file":
                with self.capture_lock:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                self.is_finished = False
                self.is_paused = False
                self.status_message = f"Lecture vidéo : {os.path.basename(self.source_path)}"

    def toggle_pause(self) -> bool:
        """Met en pause ou reprend la lecture."""
        with self.lock:
            self.is_paused = not self.is_paused
            return self.is_paused

    def get_video_progress(self) -> dict:
        """Retourne l'état d'avancement de la vidéo pour l'interface."""
        if self.source_type != "file" or self.total_frames <= 0:
            return {
                "is_video_file": False,
                "current_frame": 0,
                "total_frames": 0,
                "percent": 0.0,
                "current_time": "00:00",
                "total_time": "00:00",
                "is_finished": False,
                "is_paused": self.is_paused
            }

        cur = self.current_frame_idx
        tot = self.total_frames
        pct = min(100.0, round((cur / tot) * 100.0, 1)) if tot > 0 else 0.0

        cur_sec = int(cur / max(1.0, self.fps_source))
        tot_sec = int(tot / max(1.0, self.fps_source))

        cur_str = f"{cur_sec // 60:02d}:{cur_sec % 60:02d}"
        tot_str = f"{tot_sec // 60:02d}:{tot_sec % 60:02d}"

        return {
            "is_video_file": True,
            "current_frame": cur,
            "total_frames": tot,
            "percent": pct,
            "current_time": cur_str,
            "total_time": tot_str,
            "is_finished": self.is_finished,
            "is_paused": self.is_paused
        }

    def get_health(self) -> dict:
        """État technique compact, utilisable par le dashboard et la supervision serveur."""
        with self.lock:
            age = None
            if self.last_frame_time is not None:
                age = round(max(0.0, time.monotonic() - self.last_frame_time), 2)
            return {
                "source_type": self.source_type,
                "connected": self.is_connected,
                "status": self.status_message,
                "has_frame": self.current_frame is not None,
                "frame_age_seconds": age,
                "reconnect_attempts": self.reconnect_attempts,
                "width": self.width,
                "height": self.height,
                "source_fps": round(self.fps_source, 2),
                "finished": self.is_finished,
                "last_error": self.last_error,
            }

    def get_demo_detections(self) -> list:
        """Retourne les détections de la démo si active."""
        if self.source_type == "demo":
            return self.demo_gen.get_active_detections()
        return []

    def release(self):
        """Libère les ressources."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.capture_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
