import json
import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO


from .config import COCO_VEHICLE_CLASSES, CLASS_LABELS_FR

PERSON_CLASS_ID = 0

# Tracker imposé explicitement : depuis ultralytics 8.4, le défaut est TrackTrack, qui
# n'ouvre une piste qu'à 70 % de confiance (véhicules lointains et deux-roues jamais
# suivis) et calcule une compensation de mouvement inutile pour une caméra fixe.
# ByteTrack ouvre une piste dès 25 %.
TRACKER_CONFIG = "bytetrack.yaml"


def pick_device() -> str:
    """Carte graphique NVIDIA, puis puce Apple (M1, M2…), sinon processeur.

    ultralytics choisit seul une carte NVIDIA, mais jamais la puce graphique d'un
    Mac : sur un Mac récent, l'analyse resterait sur le processeur, plusieurs fois
    plus lente.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def engine_imgsz(model_name: str) -> Optional[int]:
    """Taille d'image fixe d'un moteur TensorRT (.engine), None pour un modèle .pt.

    Un moteur est compilé pour une seule taille : on ne peut pas faire varier la
    largeur d'analyse. ultralytics écrit ses métadonnées (dont imgsz) en tête du
    fichier : 4 octets de longueur puis du JSON.
    """
    if not str(model_name).endswith(".engine"):
        return None
    try:
        with open(model_name, "rb") as f:
            meta = json.loads(f.read(int.from_bytes(f.read(4), byteorder="little")).decode("utf-8"))
        imgsz = meta.get("imgsz", 640)
        return int(max(imgsz) if isinstance(imgsz, (list, tuple)) else imgsz)
    except Exception:
        return 640   # taille d'export utilisée par setup_jetson.sh


# Palette de couleurs Pro (Format BGR OpenCV)
CLASS_COLORS = {
    "car": (255, 180, 0),        # Bleu Cyan éclatant (B=255, G=180, R=0)
    "truck": (0, 215, 255),      # Jaune Doré néon (B=0, G=215, R=255)
    "motorcycle": (0, 255, 128), # Vert Émeraude néon (B=0, G=255, R=128)
    "bus": (255, 0, 220)         # Magenta / Violet électrique (B=255, G=0, R=220)
}

LINE_COLOR_DEFAULT = (0, 230, 255)  # Jaune / Ambre néon
LINE_COLOR_UP = (0, 255, 128)       # Vert néon (Monte)
LINE_COLOR_DOWN = (0, 140, 255)     # Orange vif (Descend)


class SimpleCentroidTracker:
    """Tracker géométrique ultra-robuste de secours (IoU + Centroïde)."""
    def __init__(self, max_disappeared: int = 25, max_distance: float = 90.0):
        self.next_id = 1
        self.objects = {}       # id -> (cx, cy, cls_name)
        self.disappeared = {}   # id -> frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def update(self, rects_with_cls: List[tuple]) -> List[tuple]:
        """
        :param rects_with_cls: [(bbox, cls_name, conf), ...]
        :return: [(id, bbox, cls_name, conf), ...]
        """
        if len(rects_with_cls) == 0:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self.objects.pop(tid, None)
                    self.disappeared.pop(tid, None)
            return []

        input_centroids = []
        for box, _, _ in rects_with_cls:
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            tracked_results = []
            for i, (box, cname, conf) in enumerate(rects_with_cls):
                tid = self.next_id
                self.next_id += 1
                self.objects[tid] = (input_centroids[i][0], input_centroids[i][1], cname)
                self.disappeared[tid] = 0
                tracked_results.append((tid, box, cname, conf))
            return tracked_results

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[tid][:2] for tid in object_ids]

        # Association par plus proche voisin
        used_rows = set()
        used_cols = set()
        matches = []

        for r, (ocx, ocy) in enumerate(object_centroids):
            for c, (icx, icy) in enumerate(input_centroids):
                dist = np.hypot(ocx - icx, ocy - icy)
                if dist < self.max_distance:
                    matches.append((dist, r, c))

        matches.sort(key=lambda x: x[0])
        assigned_results = {}

        for dist, r, c in matches:
            if r in used_rows or c in used_cols:
                continue
            tid = object_ids[r]
            box, cname, conf = rects_with_cls[c]
            self.objects[tid] = (input_centroids[c][0], input_centroids[c][1], cname)
            self.disappeared[tid] = 0
            assigned_results[c] = (tid, box, cname, conf)
            used_rows.add(r)
            used_cols.add(c)

        # Objets non assignés existants
        for r in range(len(object_ids)):
            if r not in used_rows:
                tid = object_ids[r]
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    self.objects.pop(tid, None)
                    self.disappeared.pop(tid, None)

        # Nouveaux objets apparus
        final_tracked = []
        for c in range(len(rects_with_cls)):
            if c in assigned_results:
                final_tracked.append(assigned_results[c])
            else:
                tid = self.next_id
                self.next_id += 1
                box, cname, conf = rects_with_cls[c]
                self.objects[tid] = (input_centroids[c][0], input_centroids[c][1], cname)
                self.disappeared[tid] = 0
                final_tracked.append((tid, box, cname, conf))

        return final_tracked


class YOLODetector:
    """
    Détecteur & Tracker YOLO optimisé pour les véhicules.
    Utilise YOLOv8 nano et le tracker ByteTrack avec bascule automatique sur tracker centroïde.
    """

    def __init__(self, model_name: str = "yolo11n.pt"):
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self.fallback_tracker = SimpleCentroidTracker()
        self.device = pick_device()
        self.fixed_imgsz = engine_imgsz(model_name)
        self._load_model()

    def _load_model(self):
        """Charge le modèle YOLO avec téléchargement automatique du checkpoint si nécessaire."""
        try:
            print(f"[YOLODetector] Chargement du modèle {self.model_name}...")
            self.model = YOLO(self.model_name)
            print(f"[YOLODetector] Modèle {self.model_name} prêt et opérationnel (calcul : {self.device}).")
        except Exception as e:
            print(f"[YOLODetector] Erreur lors du chargement de {self.model_name}: {e}")
            self.model = None

    def track(
        self,
        frame: np.ndarray,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        selected_classes: Optional[List[str]] = None,
        inference_width: int = 640,
        include_persons: bool = False,
    ) -> List[dict]:
        """
        Détecte et suit les véhicules dans la trame vidéo.
        
        :return: Liste de dictionnaires de détection avec IDs uniques de suivi
        """
        if self.model is None or frame is None:
            return []

        # Filtrage des classes cibles (IDs COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
        if selected_classes is None:
            target_class_ids = list(COCO_VEHICLE_CLASSES.keys())
        else:
            target_class_ids = [
                cid for cid, cname in COCO_VEHICLE_CLASSES.items()
                if cname in selected_classes
            ]

        # Les personnes servent à repérer les deux-roues que YOLO ne reconnaît pas
        # (scooters vus de face, masqués par leurs passagers) : voir core/riders.py.
        if include_persons:
            target_class_ids.append(PERSON_CLASS_ID)

        if not target_class_ids:
            return []

        # L'image d'origine est transmise telle quelle : YOLO la réduit à `imgsz` en
        # interne et renvoie les boîtes en coordonnées d'origine. Le tracker garde
        # ainsi les mêmes coordonnées si la largeur d'analyse change en direct.
        inference_frame = frame
        results = None
        if self.fixed_imgsz:
            inference_width = self.fixed_imgsz

        # 1. Tentative avec model.track (ByteTrack)
        try:
            try:
                results = self._run_track(inference_frame, conf_thresh, iou_thresh,
                                          target_class_ids, inference_width)
            except Exception as err_device:
                # Un moteur TensorRT ne tourne que sur la carte graphique.
                if self.device == "cpu" or self.fixed_imgsz:
                    raise
                # Pilote graphique absent ou opération non prise en charge : le
                # processeur reste toujours disponible.
                print(f"[YOLODetector] Échec sur {self.device} ({err_device}), passage au processeur.")
                self.device = "cpu"
                results = self._run_track(inference_frame, conf_thresh, iou_thresh,
                                          target_class_ids, inference_width)
        except Exception as err_track:
            # 2. Fallback avec model.predict si model.track rencontre un souci
            try:
                results = self.model.predict(
                    source=inference_frame,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    classes=target_class_ids,
                    verbose=False,
                    imgsz=inference_width,
                    device=self.device,
                )
            except Exception as err_pred:
                print(f"[YOLODetector] Erreur inférence: {err_pred}")
                return []

        if not results or len(results) == 0:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        # Extraction sécurisée par .tolist() (immunisée contre les conflits de version C-API NumPy)
        try:
            boxes_raw = r.boxes.xyxy.tolist() if r.boxes.xyxy is not None else []
            confs_raw = r.boxes.conf.tolist() if r.boxes.conf is not None else []
            cls_ids_raw = [int(c) for c in (r.boxes.cls.tolist() if r.boxes.cls is not None else [])]
            track_ids_raw = (
                [int(tid) for tid in r.boxes.id.tolist()]
                if r.boxes.id is not None
                else None
            )
        except Exception as e:
            print(f"[YOLODetector] Erreur extraction boîtes: {e}")
            return []

        detections = []

        # Si le tracker YOLO a retourné des IDs de suivi valides
        if track_ids_raw is not None and len(track_ids_raw) == len(boxes_raw):
            for i in range(len(boxes_raw)):
                box = [int(v) for v in boxes_raw[i]]
                conf = float(confs_raw[i])
                cls_id = cls_ids_raw[i]
                tid = track_ids_raw[i]
                cls_name = "person" if cls_id == PERSON_CLASS_ID else COCO_VEHICLE_CLASSES.get(cls_id, "car")

                detections.append({
                    "id": tid,
                    "bbox": box,
                    "class": cls_name,
                    "conf": conf
                })
        else:
            # Utilisation du tracker de secours ultra-robuste
            raw_tuples = []
            for i in range(len(boxes_raw)):
                box = [int(v) for v in boxes_raw[i]]
                conf = float(confs_raw[i])
                cls_id = cls_ids_raw[i]
                cls_name = "person" if cls_id == PERSON_CLASS_ID else COCO_VEHICLE_CLASSES.get(cls_id, "car")
                raw_tuples.append((box, cls_name, conf))

            tracked_tuples = self.fallback_tracker.update(raw_tuples)
            for tid, box, cls_name, conf in tracked_tuples:
                detections.append({
                    "id": tid,
                    "bbox": box,
                    "class": cls_name,
                    "conf": conf
                })

        return detections

    def _run_track(self, frame, conf_thresh, iou_thresh, class_ids, inference_width):
        # Demi-précision (FP16) sur carte NVIDIA. Un moteur TensorRT a déjà la sienne,
        # fixée à l'export. Paramètre omis sinon : ultralytics signale l'ancien « half ».
        precision = {}
        if self.device.startswith("cuda") and self.fixed_imgsz is None:
            precision = {"quantize": 16}
        return self.model.track(
            source=frame,
            persist=True,
            conf=conf_thresh,
            iou=iou_thresh,
            classes=class_ids,
            verbose=False,
            imgsz=inference_width,
            tracker=TRACKER_CONFIG,
            device=self.device,
            **precision,
        )

    def reset_tracker(self):
        """Réinitialise les identités lors d'un changement/replay de source."""
        self.fallback_tracker = SimpleCentroidTracker()
        if self.model is not None:
            predictor = getattr(self.model, "predictor", None)
            # Ne pas mettre `trackers` à None : avec persist=True, ultralytics ne les
            # recrée pas et chaque inférence suivante échoue (plus aucune détection).
            for tracker in getattr(predictor, "trackers", None) or []:
                tracker.reset()



def draw_direction_arrow(img: np.ndarray, x: int, baseline_y: int, up: bool,
                         color: Tuple[int, int, int], size: int = 9):
    """Triangle ▲/▼ dessiné à la main : les polices Hershey d'OpenCV n'ont que l'ASCII."""
    top = baseline_y - size
    if up:
        pts = [(x + size // 2, top), (x, baseline_y), (x + size, baseline_y)]
    else:
        pts = [(x, top), (x + size, top), (x + size // 2, baseline_y)]
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color, cv2.LINE_AA)


def draw_visuals(
    frame: np.ndarray,
    detections: List[dict],
    tracks: Dict[int, any],
    counted_ids: Dict[int, dict],
    detected_ids: Dict[int, dict],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
    stats: dict,
    show_boxes: bool = True,
    show_labels: bool = True,
    show_trails: bool = True,
    show_line: bool = True,
    show_hud: bool = True,
    speed_line_start: Tuple[int, int] = None,
    speed_line_end: Tuple[int, int] = None,
    speed_distance_m: float = 0.0,
    show_speed_line: bool = True,
) -> np.ndarray:
    """
    Rendu visuel haute définition style surveillance professionnelle (High-Tech HUD) :
    - Boîtes avec coins stylisés et couleurs par type de véhicule
    - Trajectoires fluides (trails) avec dégradé
    - Ligne laser de comptage avec indicateurs directionnels
    - Badges d'identification avec ID persistant et statut
    """
    output = frame.copy()
    h, w, _ = output.shape

    if show_speed_line and speed_line_start and speed_line_end:
        cv2.line(output, speed_line_start, speed_line_end, (255, 190, 30), 2, cv2.LINE_AA)
        label = f"VITESSE - {speed_distance_m:g} m"
        label_y = max(18, min(h - 8, speed_line_start[1] - 7))
        cv2.putText(output, label, (speed_line_start[0] + 8, label_y),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 220, 80), 1, cv2.LINE_AA)

    # 1. Dessiner la ligne de comptage laser
    if show_line:
        # Lueur externe semi-transparente
        overlay = output.copy()
        cv2.line(overlay, line_start, line_end, (0, 200, 255), 8, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)

        # Ligne centrale vive
        cv2.line(output, line_start, line_end, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Cibles d'ancrage aux extrémités
        for pt in [line_start, line_end]:
            cv2.circle(output, pt, 7, (0, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(output, pt, 11, (255, 255, 255), 1, cv2.LINE_AA)

        # Indicateurs directionnels
        mid_x = int((line_start[0] + line_end[0]) / 2)
        mid_y = int((line_start[1] + line_end[1]) / 2)
        
        # Badge MONTE (UP)
        cv2.rectangle(output, (mid_x - 75, max(5, mid_y - 28)), (mid_x + 75, max(25, mid_y - 8)), (10, 30, 15), -1)
        cv2.rectangle(output, (mid_x - 75, max(5, mid_y - 28)), (mid_x + 75, max(25, mid_y - 8)), (0, 255, 128), 1)
        draw_direction_arrow(output, mid_x - 60, max(20, mid_y - 12), True, (0, 255, 128))
        cv2.putText(output, "MONTE (UP)", (mid_x - 46, max(20, mid_y - 12)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)
        
        # Badge DESCEND (DOWN)
        cv2.rectangle(output, (mid_x - 85, min(h - 30, mid_y + 10)), (mid_x + 85, min(h - 8, mid_y + 30)), (30, 20, 10), -1)
        cv2.rectangle(output, (mid_x - 85, min(h - 30, mid_y + 10)), (mid_x + 85, min(h - 8, mid_y + 30)), (0, 140, 255), 1)
        draw_direction_arrow(output, mid_x - 75, min(h - 14, mid_y + 25), False, (0, 140, 255))
        cv2.putText(output, "DESCEND (DOWN)", (mid_x - 61, min(h - 14, mid_y + 25)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 140, 255), 1, cv2.LINE_AA)

    # 2. Dessiner les trajectoires de suivi (Motion Trails)
    if show_trails:
        for tid, trail in tracks.items():
            if len(trail) < 2:
                continue
            is_counted = tid in counted_ids
            
            pts = list(trail)
            trail_len = len(pts)
            for i in range(1, trail_len):
                # Épaisseur et intensité croissantes vers la position actuelle
                progress = i / trail_len
                thick = max(1, int(progress * 3))
                
                if is_counted:
                    direction = counted_ids[tid].get('direction', 'up')
                    base_col = (0, 255, 128) if direction == 'up' else (0, 140, 255)
                else:
                    base_col = (255, 200, 0) # Cyan/Bleu
                
                cv2.line(output, pts[i - 1], pts[i], base_col, thick, cv2.LINE_AA)

    # 3. Dessiner les boîtes de détection & Badges Pro
    for det in detections:
        bbox = det['bbox']
        tid = det.get('id')
        cls_name = det.get('class', 'car')
        conf = det.get('conf', 0.0)

        # Sélection de la couleur selon la catégorie
        theme_color = CLASS_COLORS.get(cls_name, (255, 180, 0))
        
        # Si le véhicule a déjà franchi la ligne
        is_counted = (tid is not None) and (tid in counted_ids)
        if is_counted:
            direction = counted_ids[tid].get('direction', 'up')
            box_accent_color = (0, 255, 128) if direction == 'up' else (0, 140, 255)
        else:
            box_accent_color = theme_color

        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        if show_boxes:
            # Cadre rectangulaire fin
            cv2.rectangle(output, (x1, y1), (x2, y2), box_accent_color, 1, cv2.LINE_AA)
            
            # Coins renforcés "High-Tech Bracket" (épaisseur 3px)
            corner_len = max(8, min(22, bw // 4, bh // 4))
            
            # Coin Haut-Gauche
            cv2.line(output, (x1, y1), (x1 + corner_len, y1), box_accent_color, 3, cv2.LINE_AA)
            cv2.line(output, (x1, y1), (x1, y1 + corner_len), box_accent_color, 3, cv2.LINE_AA)
            # Coin Haut-Droit
            cv2.line(output, (x2, y1), (x2 - corner_len, y1), box_accent_color, 3, cv2.LINE_AA)
            cv2.line(output, (x2, y1), (x2, y1 + corner_len), box_accent_color, 3, cv2.LINE_AA)
            # Coin Bas-Gauche
            cv2.line(output, (x1, y2), (x1 + corner_len, y2), box_accent_color, 3, cv2.LINE_AA)
            cv2.line(output, (x1, y2), (x1, y2 - corner_len), box_accent_color, 3, cv2.LINE_AA)
            # Coin Bas-Droit
            cv2.line(output, (x2, y2), (x2 - corner_len, y2), box_accent_color, 3, cv2.LINE_AA)
            cv2.line(output, (x2, y2), (x2, y2 - corner_len), box_accent_color, 3, cv2.LINE_AA)

            # Point de contact centroïde au sol / centre
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(output, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, (cx, cy), 6, box_accent_color, 1, cv2.LINE_AA)

        if show_labels:
            fr_label = CLASS_LABELS_FR.get(cls_name, cls_name.upper())
            id_str = f"#{tid}" if tid is not None else ""
            
            # Statut directionnel
            status_tag = ""
            if is_counted:
                dir_val = counted_ids[tid].get('direction')
                status_tag = " [MONTE]" if dir_val == "up" else " [DESCEND]"
                speed = counted_ids[tid].get('max_speed_kmh')
                if speed is not None:
                    status_tag += f" [{speed:.1f} km/h]"
            elif tid is not None:
                motion_direction = detected_ids.get(tid, {}).get("motion_direction")
                if motion_direction == "up":
                    status_tag = " [~MONTE]"
                elif motion_direction == "down":
                    status_tag = " [~DESCEND]"

            full_label = f"{id_str} {fr_label} {int(conf * 100)}%{status_tag}".strip()

            (tw, th), baseline = cv2.getTextSize(full_label, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
            
            # Position du badge
            badge_x1 = x1
            badge_y2 = max(th + 10, y1 - 4)
            badge_y1 = badge_y2 - th - 8
            badge_x2 = x1 + tw + 14

            # Fond noir semi-opaque pour le badge
            cv2.rectangle(output, (badge_x1, badge_y1), (badge_x2, badge_y2), (15, 15, 20), -1)
            # Bordure colorée
            cv2.rectangle(output, (badge_x1, badge_y1), (badge_x2, badge_y2), box_accent_color, 1)
            # Bandeau vertical accent
            cv2.rectangle(output, (badge_x1, badge_y1), (badge_x1 + 4, badge_y2), box_accent_color, -1)

            # Texte blanc net
            cv2.putText(output, full_label, (badge_x1 + 8, badge_y2 - 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. HUD Compact & Épuré
    if show_hud:
        # Bandeau supérieur semi-transparent
        overlay = output.copy()
        cv2.rectangle(overlay, (12, 12), (310, 88), (10, 15, 25), -1)
        cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)
        cv2.rectangle(output, (12, 12), (310, 88), (50, 65, 90), 1)

        up_cnt = stats.get("count_up", 0)
        down_cnt = stats.get("count_down", 0)
        tot_cnt = stats.get("total", 0)
        fps_val = stats.get("fps", 0.0)

        # Indicateurs
        draw_direction_arrow(output, 22, 38, True, (0, 255, 128), size=11)
        cv2.putText(output, f"MONTE: {up_cnt}", (38, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 255, 128), 1, cv2.LINE_AA)
        draw_direction_arrow(output, 160, 38, False, (0, 140, 255), size=11)
        cv2.putText(output, f"DESCEND: {down_cnt}", (176, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(output, f"TOTAL: {tot_cnt}  |  {fps_val:.1f} FPS", (22, 72),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return output
