from collections import deque
from typing import Dict, List, Tuple

# YOLO nano reconnaît mal un scooter vu de face et masqué par ses passagers, mais
# détecte bien les personnes. Une personne qui avance à vitesse de véhicule est
# donc comptée comme deux-roues. La vitesse est mesurée en « hauteurs de la
# personne par seconde » : indépendante de la distance à la caméra, sans calibration.
RIDER_MIN_SPEED_HPS = 0.9       # un piéton reste nettement en dessous
RIDER_MIN_OBSERVATIONS = 4
RIDER_MIN_DURATION_S = 0.4
RIDER_WINDOW_S = 1.5
INSIDE_VEHICLE_RATIO = 0.5      # personne vue à travers une vitre de bus, etc.
SAME_RIDE_DISTANCE = 0.6        # conducteur et passager : un seul engin


def _overlap_ratio(inner, outer) -> float:
    """Part de la boîte `inner` contenue dans `outer`."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    if w <= 0 or h <= 0:
        return 0.0
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (w * h) / area


class RiderDetector:
    """Transforme les personnes en mouvement rapide en détections « motorcycle »."""

    def __init__(self):
        self.history: Dict[int, deque] = {}
        self.riders = set()
        self.aliases: Dict[int, int] = {}   # passager -> identifiant de l'engin

    def reset(self):
        self.history.clear()
        self.riders.clear()
        self.aliases.clear()

    def process(self, detections: List[dict], timeline: float) -> List[dict]:
        """Retourne les véhicules, plus les deux-roues déduits ; les piétons sont retirés."""
        vehicles = [d for d in detections if d.get("class") != "person"]
        persons = [d for d in detections if d.get("class") == "person" and d.get("id") is not None]
        rider_dets = []
        for det in persons:
            bbox = det["bbox"]
            if any(_overlap_ratio(bbox, v["bbox"]) > INSIDE_VEHICLE_RATIO for v in vehicles):
                continue
            track_id = det["id"]
            height = max(1, bbox[3] - bbox[1])
            feet = ((bbox[0] + bbox[2]) / 2.0, float(bbox[3]))
            hist = self.history.setdefault(track_id, deque(maxlen=40))
            hist.append((timeline, feet, height))
            while hist and timeline - hist[0][0] > RIDER_WINDOW_S:
                hist.popleft()
            if track_id not in self.riders and self._moves_like_vehicle(hist):
                self.riders.add(track_id)
            if track_id in self.riders:
                rider_dets.append(det)

        return vehicles + self._merge_shared_rides(rider_dets)

    def _moves_like_vehicle(self, hist) -> bool:
        if len(hist) < RIDER_MIN_OBSERVATIONS:
            return False
        (t0, (x0, y0), _), (t1, (x1, y1), _) = hist[0], hist[-1]
        duration = t1 - t0
        if duration < RIDER_MIN_DURATION_S:
            return False
        mean_height = sum(h for _, _, h in hist) / len(hist)
        distance = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        return distance / mean_height / duration >= RIDER_MIN_SPEED_HPS

    def _merge_shared_rides(self, rider_dets: List[dict]) -> List[dict]:
        """Conducteur et passager d'un même engin ne donnent qu'un seul deux-roues."""
        kept: List[Tuple[int, dict]] = []
        for det in sorted(rider_dets, key=lambda d: self.aliases.get(d["id"], d["id"])):
            ride_id = self.aliases.get(det["id"], det["id"])
            bbox = det["bbox"]
            cx, cy, h = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, bbox[3] - bbox[1]
            same = next((k for k in kept if k[0] == ride_id or (
                abs(cx - (k[1]["bbox"][0] + k[1]["bbox"][2]) / 2) < SAME_RIDE_DISTANCE * h and
                abs(cy - (k[1]["bbox"][1] + k[1]["bbox"][3]) / 2) < SAME_RIDE_DISTANCE * h)), None)
            if same is not None:
                self.aliases[det["id"]] = same[0]
                box = same[1]["bbox"]
                same[1]["bbox"] = [min(box[0], bbox[0]), min(box[1], bbox[1]),
                                   max(box[2], bbox[2]), max(box[3], bbox[3])]
                continue
            kept.append((ride_id, {**det, "id": ride_id, "class": "motorcycle", "bbox": list(bbox)}))
        return [d for _, d in kept]
