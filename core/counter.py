import time
import threading
from typing import Callable, Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime



def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> float:
    """Calcul du produit vectoriel 2D (Counter-Clockwise / orientation)."""
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])


def intersect(A: Tuple[float, float], B: Tuple[float, float], 
              C: Tuple[float, float], D: Tuple[float, float]) -> bool:
    """Vérifie si le segment AB coupe le segment CD."""
    d1 = ccw(A, B, C)
    d2 = ccw(A, B, D)
    d3 = ccw(C, D, A)
    d4 = ccw(C, D, B)
    
    # Convention demi-ouverte : un point posé exactement sur la ligne compte du
    # côté positif. Sinon un centre tombant pile sur la ligne ne coupe ni le
    # segment qui y arrive ni celui qui en repart, et le véhicule passe inaperçu.
    return (d1 >= 0) != (d2 >= 0) and (d3 >= 0) != (d4 >= 0)


# Pourquoi un véhicule détecté n'a pas été compté (aucun franchissement de ligne)
UNCOUNTED_REASONS = {
    "already_past": "déjà au-delà de la ligne à son apparition",
    "exited_edge": "sorti par un bord de l'image avant la ligne",
    "brief": "vu trop brièvement (doublon d'identifiant ou fausse détection probable)",
    "lost": "perdu de vue avant la ligne (masqué ou changement d'identifiant)",
    "still_visible": "encore dans l'image, pas encore arrivé à la ligne",
}
BRIEF_TRACK_FRAMES = 4
# Une détection n'est retenue qu'après avoir été suivie un minimum de temps
# vidéo : à 25 i/s, quelques images ne durent qu'une fraction de seconde et
# laissent passer reflets et fausses détections ; en direct à faible débit,
# une règle en images seules serait au contraire beaucoup plus exigeante.
MIN_CONFIRM_SECONDS = 0.6

# Raccord des fragments : le tracker donne parfois un nouveau numéro à un véhicule
# perdu un instant (occultation, détection manquée). Un nouveau numéro qui apparaît
# là où un véhicule récemment perdu devait se trouver est rattaché à ce véhicule.
FRAGMENT_MAX_GAP_S = 1.5
FRAGMENT_DIST_RATIO = 0.8       # distance tolérée, en diagonales de la boîte
FRAGMENT_AREA_RATIO = (0.4, 2.5)
EDGE_MARGIN_RATIO = 0.02

# Un véhicule au sens établi qui disparaît sans avoir franchi la ligne (sorti par
# le haut ou le bas, perdu au loin) est compté dans ce sens. On attend plus que le
# délai de raccord des fragments : s'il réapparaît sous un autre numéro, il est
# d'abord rattaché à sa piste et ne peut pas être compté deux fois.
EXIT_COUNT_S = 2.0
EXIT_COUNT_FRAMES = 5


class VehicleCounter:
    """
    Système robuste de comptage directionnel de véhicules.
    Gère le franchissement de ligne virtuelle, l'orientation (monte/descend),
    l'historique des trajectoires et les statistiques en temps réel.
    """

    def __init__(self, max_trail_len: int = 30):
        self.max_trail_len = max_trail_len
        
        # Trajectoires: track_id -> deque([(x, y), ...])
        self.tracks: Dict[int, deque] = {}
        self.track_classes: Dict[int, str] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.track_directions: Dict[int, str] = {}
        
        # ID déjà comptabilisés: track_id -> event dict
        self.counted_ids: Dict[int, dict] = {}
        self.pending_crossings: Dict[int, dict] = {}
        self.detected_ids: Dict[int, dict] = {}
        self.detection_hits: Dict[int, int] = {}
        self.detection_start: Dict[int, float] = {}
        self.vehicle_speeds: Dict[int, float] = {}
        self.speed_last_observation: Dict[int, Tuple[Tuple[int, int], float]] = {}
        self.speed_samples: Dict[int, deque] = {}
        # Diagnostic des non-comptés : première position, dernière boîte, images vues
        self.track_first_pt: Dict[int, Tuple[int, int]] = {}
        self.track_last_bbox: Dict[int, list] = {}
        self.track_frames: Dict[int, int] = {}
        self.frame_size: Optional[Tuple[int, int]] = None
        self.track_aliases: Dict[int, int] = {}          # numéro du tracker -> véhicule
        self.track_last_time: Dict[int, float] = {}
        self.track_velocity: Dict[int, Tuple[float, float]] = {}  # px/s
        self.track_class_votes: Dict[int, Dict[str, int]] = {}
        self.last_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        
        # Compteurs globaux
        self.count_up: int = 0      # Véhicules qui montent (⬆️ Nord / Entrant)
        self.count_down: int = 0    # Véhicules qui descendent (⬇️ Sud / Sortant)
        
        # Compteurs par catégorie
        self.counts_by_class: Dict[str, Dict[str, int]] = {
            "car": {"up": 0, "down": 0, "total": 0},
            "motorcycle": {"up": 0, "down": 0, "total": 0},
            "bus": {"up": 0, "down": 0, "total": 0},
            "truck": {"up": 0, "down": 0, "total": 0}
        }
        
        # Historique des événements récents pour le dashboard
        self.recent_events: deque = deque(maxlen=30)
        # Historique complet pour l'export CSV (une caméra peut tourner des jours :
        # borné, environ 25 Mo au maximum).
        self.event_log: deque = deque(maxlen=50000)
        
        # Débit glissant (horodatages des passages sur les 60 dernières secondes)
        self.crossing_timestamps: deque = deque(maxlen=500)
        
        # Métriques de performance
        self.fps: float = 0.0
        # Verrou de synchronisation inter-threads
        self.lock = threading.RLock()
        self.start_time = time.time()
        self.total_frames_processed = 0
        self.current_timeline = 0.0

        # Agrégats des véhicules sortis de la mémoire active (flux 24h/24)
        self._reset_archive()

    def _reset_archive(self):
        self.archived_total = 0
        self.archived_awaiting_crossing = 0
        self.archived_awaiting_direction = 0
        self.archived_direction_known = 0
        self.archived_speed_sum = 0.0
        self.archived_speed_count = 0
        self.archived_speed_max = 0.0
        self.archived_uncounted = {reason: 0 for reason in UNCOUNTED_REASONS}

    def reset(self):
        """Réinitialise tous les compteurs et l'historique."""
        with self.lock:
            self.tracks.clear()
            self.track_classes.clear()
            self.track_last_seen.clear()
            self.track_directions.clear()
            self.counted_ids.clear()
            self.pending_crossings.clear()
            self.detected_ids.clear()
            self.detection_hits.clear()
            self.detection_start.clear()
            self.vehicle_speeds.clear()
            self.speed_last_observation.clear()
            self.speed_samples.clear()
            self.track_first_pt.clear()
            self.track_last_bbox.clear()
            self.track_frames.clear()
            self.track_aliases.clear()
            self.track_last_time.clear()
            self.track_velocity.clear()
            self.track_class_votes.clear()
            self.count_up = 0
            self.count_down = 0
            for cls_name in self.counts_by_class:
                self.counts_by_class[cls_name] = {"up": 0, "down": 0, "total": 0}
            self.recent_events.clear()
            self.event_log.clear()
            self.crossing_timestamps.clear()
            self.start_time = time.time()
            self.total_frames_processed = 0
            self.current_timeline = 0.0
            self._reset_archive()

    def update_tracks(
        self,
        detections: List[dict],
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        buffer_px: int = 20,
        timeline: float = None,
        confirm_frames: int = 2,
        speed_line_start: Tuple[int, int] = None,
        speed_line_end: Tuple[int, int] = None,
        speed_distance_m: float = 0.0,
        ground_distance: Optional[Callable[[float], float]] = None,
        frame_size: Optional[Tuple[int, int]] = None,
        merge_fragments: bool = False,
        count_exits: bool = False,
    ):
        """
        Met à jour les trajectoires et évalue le franchissement de la ligne.
        """
        with self.lock:
            self.frame_size = frame_size
            self.last_line = (line_start, line_end)
            self.total_frames_processed += 1
            frame_number = self.total_frames_processed
            self.current_timeline = float(timeline if timeline is not None else time.monotonic())
            if merge_fragments:
                detections = self._merge_fragments(detections)
            seen_ids = set()
            for det in detections:
                track_id = det.get('id')
                bbox = det.get('bbox')
                if track_id is None or not bbox or len(bbox) != 4:
                    continue
                seen_ids.add(track_id)

                cls_name = det.get('class', 'car')
                if cls_name not in self.counts_by_class:
                    cls_name = 'car'
                # Catégorie = la plus souvent vue sur toute la piste : YOLO hésite
                # souvent entre bus, camion et voiture d'une image à l'autre.
                votes = self.track_class_votes.setdefault(track_id, {})
                votes[cls_name] = votes.get(cls_name, 0) + 1
                cls_name = max(votes, key=votes.get)
                event = self.detected_ids.get(track_id)
                if event is not None and event["class"] != cls_name:
                    self._reclassify(event, cls_name)

            # Calcul du centre du véhicule (centre géométrique)
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                curr_pt = (cx, cy)

                self.track_classes[track_id] = cls_name
                self.track_last_seen[track_id] = frame_number
                self.detection_hits[track_id] = self.detection_hits.get(track_id, 0) + 1
                if self.detection_hits[track_id] == 1:
                    self.detection_start[track_id] = self.current_timeline

                if (track_id not in self.detected_ids and
                        self.detection_hits[track_id] >= max(1, confirm_frames) and
                        self.current_timeline - self.detection_start[track_id] >= MIN_CONFIRM_SECONDS):
                    self._record_detection(track_id, cls_name)

                if track_id not in self.tracks:
                    self.tracks[track_id] = deque(maxlen=self.max_trail_len)
                    self.track_first_pt[track_id] = curr_pt
                last_time = self.track_last_time.get(track_id)
                last_box = self.track_last_bbox.get(track_id)
                if last_time is not None and last_box and self.current_timeline > last_time:
                    dt = self.current_timeline - last_time
                    vx = ((bbox[0] + bbox[2]) - (last_box[0] + last_box[2])) / 2 / dt
                    vy = ((bbox[1] + bbox[3]) - (last_box[1] + last_box[3])) / 2 / dt
                    pvx, pvy = self.track_velocity.get(track_id, (vx, vy))
                    self.track_velocity[track_id] = (0.6 * pvx + 0.4 * vx, 0.6 * pvy + 0.4 * vy)
                self.track_last_time[track_id] = self.current_timeline
                self.track_last_bbox[track_id] = bbox
                self.track_frames[track_id] = self.track_frames.get(track_id, 0) + 1
            
                trail = self.tracks[track_id]
                previous_pt = trail[-1] if trail else None
                trail.append(curr_pt)

                stable_direction = self._infer_stable_direction(
                    trail, line_start, line_end, buffer_px
                )
                if stable_direction is not None:
                    self.track_directions[track_id] = stable_direction
                    event = self.detected_ids.get(track_id)
                    if event is not None:
                        event["motion_direction"] = stable_direction

                if (speed_line_start is not None and speed_line_end is not None and
                        speed_distance_m > 0):
                    # Avec la correction de perspective, on mesure au point de contact
                    # avec la route (bas de la boîte), seul point réellement au sol.
                    speed_pt = (cx, int(bbox[3])) if ground_distance else curr_pt
                    self._update_speed(
                        track_id, speed_pt,
                        line_start, line_end, speed_line_start, speed_line_end,
                        speed_distance_m, ground_distance,
                    )

                # Le franchissement et la direction sont deux décisions distinctes.
                # On attend un mouvement stable avant de renseigner Monte/Descend.
                if track_id not in self.counted_ids:
                    if (track_id not in self.pending_crossings and len(trail) >= 2 and
                            self._has_line_crossed(trail, line_start, line_end)):
                        self.pending_crossings[track_id] = {
                            "class": cls_name,
                            "timeline": self.current_timeline,
                            "frame": frame_number,
                        }
                        event = self.detected_ids.get(track_id)
                        if event is not None:
                            event["status"] = "direction_pending"
                    if (track_id in self.pending_crossings and
                            track_id in self.track_directions):
                        self._record_crossing(
                            track_id, cls_name, self.track_directions[track_id]
                        )
                        self.pending_crossings.pop(track_id, None)
                    if (track_id in self.pending_crossings and
                            frame_number - self.pending_crossings[track_id]["frame"] >= 2):
                        direction = self._infer_stable_direction(
                            trail, line_start, line_end, buffer_px
                        )
                        if direction is not None:
                            self._record_crossing(track_id, cls_name, direction)
                            self.pending_crossings.pop(track_id, None)

            for track_id in list(self.detection_hits):
                if track_id not in seen_ids and track_id not in self.detected_ids:
                    self.detection_hits[track_id] = 0

            if count_exits:
                for tid in list(self.detected_ids):
                    last_time = self.track_last_time.get(tid)
                    if (tid not in seen_ids and last_time is not None and
                            self.current_timeline - last_time >= EXIT_COUNT_S and
                            frame_number - self.track_last_seen.get(tid, frame_number) >= EXIT_COUNT_FRAMES):
                        self._count_exit(tid)

            # L'âge est mesuré en trames et non en secondes murales. Une inférence
            # CPU lente ne doit jamais effacer une trajectoire vidéo encore valide.
            expired_ids = [
                tid for tid, last_frame in self.track_last_seen.items()
                if frame_number - last_frame > 90
            ]
            for tid in expired_ids:
                if count_exits:
                    self._count_exit(tid)
                event = self.detected_ids.get(tid)
                if event is not None and event.get("status") == "before_line":
                    event["uncounted_reason"] = self._uncounted_reason(tid, final=True)
                self.track_first_pt.pop(tid, None)
                self.track_last_bbox.pop(tid, None)
                self.track_frames.pop(tid, None)
                self.track_last_time.pop(tid, None)
                self.track_velocity.pop(tid, None)
                self.track_class_votes.pop(tid, None)
                for raw in [r for r, vehicle in self.track_aliases.items() if vehicle == tid]:
                    del self.track_aliases[raw]
                self.tracks.pop(tid, None)
                self.track_classes.pop(tid, None)
                self.track_directions.pop(tid, None)
                self.track_last_seen.pop(tid, None)
                self.detection_hits.pop(tid, None)
                self.detection_start.pop(tid, None)
                self.speed_last_observation.pop(tid, None)
                self.speed_samples.pop(tid, None)
                self.pending_crossings.pop(tid, None)
                # Le tracker ne réutilise pas un ID disparu depuis 90 trames :
                # sa détection est définitive et peut être résumée en agrégats.
                self.counted_ids.pop(tid, None)
                self._archive_vehicle(tid)

    def _merge_fragments(self, detections: List[dict]) -> List[dict]:
        """Renumérote les détections : un fragment reprend le numéro de son véhicule."""
        out, fresh, present = [], [], set()
        for det in detections:
            raw = det.get('id')
            if raw is None:
                out.append(det)
            elif raw in self.track_aliases or raw in self.tracks:
                vehicle = self.track_aliases.get(raw, raw)
                present.add(vehicle)
                out.append({**det, 'id': vehicle})
            else:
                fresh.append(det)
        for det in fresh:
            vehicle = self._find_lost_vehicle(det['bbox'], present)
            if vehicle is None:
                out.append(det)
                continue
            self.track_aliases[det['id']] = vehicle
            present.add(vehicle)
            out.append({**det, 'id': vehicle})
        return out

    def _find_lost_vehicle(self, bbox, present) -> Optional[int]:
        """Véhicule perdu récemment dont la position prédite correspond à `bbox`."""
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        best, best_dist = None, None
        for tid, last_time in self.track_last_time.items():
            gap = self.current_timeline - last_time
            if tid in present or not 0 < gap <= FRAGMENT_MAX_GAP_S:
                continue
            last = self.track_last_bbox.get(tid)
            if not last:
                continue
            w, h = last[2] - last[0], last[3] - last[1]
            if not FRAGMENT_AREA_RATIO[0] <= area / max(1, w * h) <= FRAGMENT_AREA_RATIO[1]:
                continue
            vx, vy = self.track_velocity.get(tid, (0.0, 0.0))
            px = (last[0] + last[2]) / 2 + vx * gap
            py = (last[1] + last[3]) / 2 + vy * gap
            dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            if dist <= max(20.0, FRAGMENT_DIST_RATIO * (w * w + h * h) ** 0.5) and (
                    best_dist is None or dist < best_dist):
                best, best_dist = tid, dist
        return best

    def _reclassify(self, event: dict, new_class: str):
        old_class = event["class"]
        self.counts_by_class[old_class]["total"] -= 1
        self.counts_by_class[new_class]["total"] += 1
        direction = event.get("direction")
        if direction in ("up", "down"):
            self.counts_by_class[old_class][direction] -= 1
            self.counts_by_class[new_class][direction] += 1
        event["class"] = new_class

    def _uncounted_reason(self, track_id: int, final: bool) -> str:
        """Explique pourquoi un véhicule n'a jamais franchi la ligne.

        `final` : le véhicule a disparu définitivement (sinon il est encore suivi).
        """
        direction = self.track_directions.get(track_id) or (
            self.detected_ids.get(track_id, {}).get("motion_direction"))
        first = self.track_first_pt.get(track_id)
        if direction in ("up", "down") and first is not None and self.last_line:
            (x1, y1), (x2, y2) = self.last_line
            horizontal = abs(x2 - x1) >= abs(y2 - y1)
            value = first[1] if horizontal else first[0]
            # Position de la ligne à l'abscisse (ou l'ordonnée) du véhicule
            if horizontal:
                line_value = y1 + (y2 - y1) * ((first[0] - x1) / (x2 - x1) if x2 != x1 else 0)
            else:
                line_value = x1 + (x2 - x1) * ((first[1] - y1) / (y2 - y1) if y2 != y1 else 0)
            # « down » = coordonnée croissante : au-delà de la ligne si déjà plus grande
            if (direction == "down") == (value > line_value):
                return "already_past"
        if not final:
            return "still_visible"
        if self.track_frames.get(track_id, 0) < BRIEF_TRACK_FRAMES:
            return "brief"
        bbox = self.track_last_bbox.get(track_id)
        if bbox and self.frame_size:
            width, height = self.frame_size
            mx, my = width * EDGE_MARGIN_RATIO, height * EDGE_MARGIN_RATIO
            if bbox[0] <= mx or bbox[1] <= my or bbox[2] >= width - mx or bbox[3] >= height - my:
                return "exited_edge"
        return "lost"

    def _has_line_crossed(self, trail, line_start, line_end) -> bool:
        """Détecte le franchissement sans supposer immédiatement une direction."""
        pts = list(trail)
        if len(pts) < 2:
            return False
        return intersect(pts[-2], pts[-1], line_start, line_end)

    def _infer_stable_direction(self, trail, line_start, line_end, buffer_px):
        """Retourne une direction seulement après un mouvement cohérent sur plusieurs trames."""
        pts = list(trail)[-10:]
        if len(pts) < 4:
            return None
        horizontal = abs(line_end[0] - line_start[0]) >= abs(line_end[1] - line_start[1])
        values = [point[1] if horizontal else point[0] for point in pts]
        deltas = [b - a for a, b in zip(values, values[1:]) if abs(b - a) >= 1]
        if len(deltas) < 3:
            return None

        displacement = values[-1] - values[0]
        minimum_displacement = max(8, int(buffer_px * 0.6))
        if abs(displacement) < minimum_displacement:
            return None
        expected_positive = displacement > 0
        consistent = sum((delta > 0) == expected_positive for delta in deltas) / len(deltas)
        if consistent < 0.75:
            return None
        return "down" if expected_positive else "up"


    def _count_exit(self, track_id: int):
        """Compte un véhicule disparu avant la ligne, si son sens de circulation est établi."""
        event = self.detected_ids.get(track_id)
        if event is None or event.get("status") != "before_line":
            return
        direction = self.track_directions.get(track_id) or event.get("motion_direction")
        if direction in ("up", "down"):
            self._record_crossing(track_id, event["class"], direction, counted_by="exit")

    def _record_crossing(self, track_id: int, cls_name: str, direction: str, counted_by: str = "line"):
        """Enregistre le franchissement valide et met à jour les statistiques."""
        now = time.time()
        
        if direction == "up":
            self.count_up += 1
            self.counts_by_class[cls_name]["up"] += 1
        else:
            self.count_down += 1
            self.counts_by_class[cls_name]["down"] += 1
            
        if track_id not in self.detected_ids:
            self._record_detection(track_id, cls_name)

        event = self.detected_ids[track_id]
        event["direction"] = direction
        event["motion_direction"] = direction
        event["status"] = "directed"
        event["counted_by"] = counted_by
        event["direction_timestamp"] = now
        event["time_str"] = datetime.now().strftime("%H:%M:%S")
        
        self.counted_ids[track_id] = event
        self.crossing_timestamps.append(self.current_timeline)

    def _record_detection(self, track_id: int, cls_name: str):
        """Ajoute une détection confirmée, avant même de connaître sa direction."""
        event = {
            "id": track_id,
            "class": cls_name,
            "direction": None,
            "motion_direction": None,
            "status": "before_line",
            "timestamp": time.time(),
            "time_str": datetime.now().strftime("%H:%M:%S"),
            "max_speed_kmh": self.vehicle_speeds.get(track_id),
            "counted_by": None,   # "line" : ligne franchie ; "exit" : disparu dans un sens établi
            "identity": None,   # marque / modèle, rempli plus tard par VehicleIdentifier
        }
        self.detected_ids[track_id] = event
        self.counts_by_class[cls_name]["total"] += 1
        self.recent_events.appendleft(event)
        self.event_log.append(event)

    def _update_speed(
        self, track_id, current_pt,
        line_a_start, line_a_end, line_b_start, line_b_end,
        distance_m: float,
        ground_distance: Optional[Callable[[float], float]] = None,
    ):
        """Estime la vitesse dans la zone calibrée avec une médiane anti-jitter."""
        previous = self.speed_last_observation.get(track_id)
        self.speed_last_observation[track_id] = (current_pt, self.current_timeline)
        if previous is None:
            return
        previous_pt, previous_time = previous
        duration = self.current_timeline - previous_time
        if duration <= 0:
            return

        horizontal = abs(line_a_end[0] - line_a_start[0]) >= abs(line_a_end[1] - line_a_start[1])
        axis = 1 if horizontal else 0
        line_a_pos = (line_a_start[axis] + line_a_end[axis]) / 2.0
        line_b_pos = (line_b_start[axis] + line_b_end[axis]) / 2.0
        pixel_distance = abs(line_b_pos - line_a_pos)
        if pixel_distance < 5:
            return

        zone_min, zone_max = sorted((line_a_pos, line_b_pos))
        current_axis = current_pt[axis]
        previous_axis = previous_pt[axis]
        margin = pixel_distance * 0.08
        if not (zone_min - margin <= current_axis <= zone_max + margin or
                zone_min - margin <= previous_axis <= zone_max + margin):
            return

        if ground_distance is not None and horizontal:
            # Perspective : un pixel couvre plus de route en haut de l'image qu'en bas.
            moved_m = abs(ground_distance(current_axis) - ground_distance(previous_axis))
        else:
            moved_m = abs(current_axis - previous_axis) / pixel_distance * distance_m
        speed = moved_m / duration * 3.6
        if not 0.5 <= speed <= 350.0:
            return
        samples = self.speed_samples.setdefault(track_id, deque(maxlen=7))
        samples.append(speed)
        if len(samples) < 3:
            return
        ordered = sorted(samples)
        stable_speed = ordered[len(ordered) // 2]
        rounded_speed = round(stable_speed, 1)
        self.vehicle_speeds[track_id] = max(self.vehicle_speeds.get(track_id, 0.0), rounded_speed)
        event = self.detected_ids.get(track_id)
        if event is not None:
            event["max_speed_kmh"] = self.vehicle_speeds[track_id]

    def _archive_vehicle(self, track_id: int):
        """Retire un véhicule terminé de la mémoire en conservant ses statistiques."""
        event = self.detected_ids.pop(track_id, None)
        if event is not None:
            self.archived_total += 1
            status = event.get("status")
            if status == "before_line":
                self.archived_awaiting_crossing += 1
            elif status == "direction_pending":
                self.archived_awaiting_direction += 1
            if event.get("motion_direction") in ("up", "down"):
                self.archived_direction_known += 1
            reason = event.get("uncounted_reason")
            if reason in self.archived_uncounted:
                self.archived_uncounted[reason] += 1
        speed = self.vehicle_speeds.pop(track_id, None)
        if speed is not None:
            self.archived_speed_sum += speed
            self.archived_speed_count += 1
            self.archived_speed_max = max(self.archived_speed_max, speed)

    def get_event_log(self) -> List[dict]:
        """Tous les véhicules détectés depuis la dernière remise à zéro, du plus ancien au plus récent."""
        with self.lock:
            return [dict(ev) for ev in self.event_log]

    def get_flow_rate_per_minute(self) -> float:
        """Calcule le débit de véhicules par minute sur la dernière minute glissante."""
        now = self.current_timeline
        # Filtrer les horodatages dans les 60 dernières secondes
        while self.crossing_timestamps and now - self.crossing_timestamps[0] > 60.0:
            self.crossing_timestamps.popleft()
        return float(len(self.crossing_timestamps))

    def get_stats(self) -> dict:
        """Retourne l'état complet des statistiques pour l'API et le dashboard."""
        with self.lock:
            total = self.archived_total + len(self.detected_ids)
            pending = max(0, total - self.count_up - self.count_down)
            awaiting_direction = self.archived_awaiting_direction + sum(
                1 for event in self.detected_ids.values()
                if event.get("status") == "direction_pending"
            )
            awaiting_crossing = self.archived_awaiting_crossing + sum(
                1 for event in self.detected_ids.values()
                if event.get("status") == "before_line"
            )
            direction_known = self.archived_direction_known + sum(
                1 for event in self.detected_ids.values()
                if event.get("motion_direction") in ("up", "down")
            )
            flow_rate = self.get_flow_rate_per_minute()
            speeds = list(self.vehicle_speeds.values())
            speed_count = self.archived_speed_count + len(speeds)
            speed_sum = self.archived_speed_sum + sum(speeds)
            speed_max = max(speeds + [self.archived_speed_max]) if speed_count else 0.0
            
            return {
                "count_up": self.count_up,
                "count_down": self.count_down,
                "total": total,
                "count_pending": pending,
                "count_awaiting_crossing": awaiting_crossing,
                "count_awaiting_direction": awaiting_direction,
                "direction_known_count": direction_known,
                "direction_unknown_count": max(0, total - direction_known),
                "flow_rate_vpm": round(flow_rate, 1),
                "fps": round(self.fps, 1),
                "max_speed_kmh": round(speed_max, 1),
                "avg_speed_kmh": round(speed_sum / speed_count, 1) if speed_count else 0.0,
                "speed_vehicles_count": speed_count,
                "by_class": {name: values.copy() for name, values in self.counts_by_class.items()},
                "recent_events": list(self.recent_events),
                "active_tracks_count": len(self.tracks),
                "uncounted_reasons": self._uncounted_summary(),
            }

    def _uncounted_summary(self) -> List[dict]:
        """Répartition des véhicules détectés mais jamais comptés, par cause."""
        counts = dict(self.archived_uncounted)
        for track_id, event in self.detected_ids.items():
            if event.get("status") != "before_line":
                continue
            reason = event.get("uncounted_reason")
            if reason is None:
                # Encore suivi : seule « déjà passé » est déjà certaine
                reason = self._uncounted_reason(track_id, final=False)
            counts[reason] = counts.get(reason, 0) + 1
        return [
            {"reason": key, "label": label, "count": counts[key]}
            for key, label in UNCOUNTED_REASONS.items() if counts.get(key)
        ]

    def get_visual_state(self):
        """Copie cohérente des données nécessaires au rendu."""
        with self.lock:
            tracks = {tid: deque(points, maxlen=points.maxlen) for tid, points in self.tracks.items()}
            return tracks, dict(self.counted_ids), dict(self.detected_ids)
