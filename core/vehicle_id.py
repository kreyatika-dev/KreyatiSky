import queue
import threading
import time
from collections import Counter, deque
from typing import Dict, Literal, Optional

import cv2
from pydantic import BaseModel

# Identification de la marque et du modèle par l'API Claude (vision).
# Seuls les véhicules comptés sont envoyés, une seule fois chacun, avec la plus
# grande image vue pendant leur passage : c'est là que les détails sont lisibles.
CROP_MARGIN = 0.10              # marge autour de la boîte, en part de sa taille
CROP_MAX_SIDE = 640             # suffisant pour la calandre et les logos
CROP_MIN_WIDTH = 40             # en dessous, rien n'est lisible : inutile de payer
CROP_GROWTH = 1.15              # n'encode une nouvelle image que si elle est nettement plus grande
SEND_DELAY_S = 2.0              # temps laissé au véhicule pour se rapprocher après la ligne
QUEUE_SIZE = 50

SYSTEM_PROMPT = (
    "Tu identifies des véhicules sur des images de caméra de circulation. "
    "L'image est un recadrage autour d'un seul véhicule, souvent petit, flou ou vu de dos. "
    "Donne la marque et le modèle seulement si des indices visibles les soutiennent "
    "(logo, calandre, phares, silhouette caractéristique). Si tu hésites entre plusieurs "
    "modèles, donne la marque seule et mets « inconnu » pour le modèle ; si même la marque "
    "est incertaine, mets « inconnu » pour les deux. Une réponse « inconnu » honnête vaut "
    "mieux qu'une supposition. Réponds en français pour la couleur et le type de carrosserie."
)


class VehicleIdentity(BaseModel):
    make: str
    model: str
    color: str
    body_type: str
    confidence: Literal["haute", "moyenne", "faible"]


IDENTITY_SCHEMA = {**VehicleIdentity.model_json_schema(), "additionalProperties": False}


class VehicleIdentifier:
    """File d'attente d'identification, traitée par un thread dédié pour ne jamais ralentir l'analyse vidéo."""

    def __init__(self, config_manager, counter, client_factory=None):
        self.config_manager = config_manager
        self.counter = counter
        self._client_factory = client_factory or self._default_client
        self._client = None
        self._client_key = None
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_SIZE)
        self._lock = threading.Lock()
        self._best: Dict[int, tuple] = {}       # véhicule -> (surface, jpeg)
        self._waiting: Dict[int, float] = {}    # véhicule compté -> instant du comptage
        self._sent = set()
        self._generation = 0
        self._sent_times: deque = deque()
        self.identified = 0
        self.failed = 0
        self.skipped = 0
        self.last_error: Optional[str] = None
        self._makes: Counter = Counter()
        self._thread = threading.Thread(target=self._worker, name="vehicle-id", daemon=True)
        self._thread.start()

    def reset(self):
        with self._lock:
            self._generation += 1
            self._best.clear()
            self._waiting.clear()
            self._sent.clear()
            self._makes.clear()
            self.identified = self.failed = self.skipped = 0
            self.last_error = None
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    # --- Thread d'analyse vidéo -------------------------------------------------

    def observe(self, frame, detections, timeline: float, cfg):
        """Appelé après chaque image analysée : garde les meilleures images, envoie les véhicules comptés."""
        if not cfg.vehicle_id_enabled:
            return
        counter = self.counter
        with counter.lock:
            aliases = dict(counter.track_aliases)
            confirmed = set(counter.detected_ids)
            counted = dict(counter.counted_ids)
            active = set(counter.tracks)

        height, width = frame.shape[:2]
        with self._lock:
            for det in detections:
                raw = det.get("id")
                if raw is None:
                    continue
                vehicle = aliases.get(raw, raw)
                if vehicle not in confirmed or vehicle in self._sent:
                    continue
                x1, y1, x2, y2 = det["bbox"]
                area = max(0, x2 - x1) * max(0, y2 - y1)
                best = self._best.get(vehicle)
                if (x2 - x1) < CROP_MIN_WIDTH or (best and area < best[0] * CROP_GROWTH):
                    continue
                jpeg = self._encode_crop(frame, (x1, y1, x2, y2), width, height)
                if jpeg is not None:
                    self._best[vehicle] = (area, jpeg)

            for vehicle in counted:
                if vehicle not in self._sent and vehicle not in self._waiting:
                    self._waiting[vehicle] = timeline

            ready = [v for v, t in self._waiting.items()
                     if v not in active or timeline - t >= SEND_DELAY_S]
            for vehicle in ready:
                self._waiting.pop(vehicle, None)
                self._sent.add(vehicle)
                best = self._best.pop(vehicle, None)
                event = counted.get(vehicle)
                if event is None:
                    continue
                if best is None:
                    self._set_identity(event, {"status": "too_small"})
                    self.skipped += 1
                elif not self._within_rate(cfg.vehicle_id_max_per_minute):
                    self._set_identity(event, {"status": "quota"})
                    self.skipped += 1
                else:
                    # « En attente » d'abord : le thread d'identification peut répondre
                    # avant le retour de put_nowait et ne doit pas être écrasé.
                    self._set_identity(event, {"status": "pending"})
                    try:
                        self._queue.put_nowait((self._generation, event, best[1]))
                    except queue.Full:
                        self._set_identity(event, {"status": "quota"})
                        self.skipped += 1

            # Véhicules confirmés mais jamais comptés : leur image ne servira pas.
            for vehicle in [v for v in self._best if v not in active]:
                del self._best[vehicle]

    def _within_rate(self, per_minute: int) -> bool:
        now = time.monotonic()
        while self._sent_times and now - self._sent_times[0] > 60.0:
            self._sent_times.popleft()
        if len(self._sent_times) >= per_minute:
            return False
        self._sent_times.append(now)
        return True

    @staticmethod
    def _encode_crop(frame, bbox, width, height) -> Optional[bytes]:
        x1, y1, x2, y2 = bbox
        mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
        x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
        x2, y2 = min(width, int(x2 + mx)), min(height, int(y2 + my))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        scale = CROP_MAX_SIDE / max(crop.shape[:2])
        if scale < 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return encoded.tobytes() if ok else None

    def _set_identity(self, event: dict, identity: dict):
        # Remplacement d'une clé existante : le JSON des statistiques peut être en
        # cours de sérialisation dans un autre thread.
        with self.counter.lock:
            event["identity"] = identity

    # --- Thread d'identification -------------------------------------------------

    @staticmethod
    def _default_client(api_key: str):
        import anthropic
        # Sans clé saisie dans le dashboard, le SDK utilise ANTHROPIC_API_KEY ou le profil `ant auth login`.
        return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def _get_client(self, api_key: str):
        if self._client is None or api_key != self._client_key:
            self._client = self._client_factory(api_key)
            self._client_key = api_key
        return self._client

    def _worker(self):
        # Chargement du SDK en tâche de fond, dès le démarrage : il prend plusieurs
        # secondes et ne doit retarder ni le serveur ni la première identification.
        try:
            import anthropic  # noqa: F401
        except ImportError:
            pass
        while True:
            generation, event, jpeg = self._queue.get()
            cfg = self.config_manager.get()
            if generation != self._generation or not cfg.vehicle_id_enabled:
                continue
            identity = self._identify(cfg, event.get("class", "car"), jpeg)
            with self._lock:
                if generation != self._generation:
                    continue
                if identity.get("status") == "done":
                    self.identified += 1
                    if identity["make"].lower() != "inconnu":
                        self._makes[identity["make"]] += 1
                else:
                    self.failed += 1
            self._set_identity(event, identity)

    def _identify(self, cfg, cls_name: str, jpeg: bytes) -> dict:
        import base64
        import anthropic

        try:
            client = self._get_client(cfg.vehicle_id_api_key)
            response = client.beta.messages.create(
                model=cfg.vehicle_id_model,
                max_tokens=4096,
                output_config={
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": IDENTITY_SCHEMA},
                },
                # Si le modèle décline, l'API relance la même demande sur un modèle de repli.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.standard_b64encode(jpeg).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": f"Catégorie détectée par le compteur : {cls_name}. Identifie ce véhicule."},
                    ],
                }],
            )
        except anthropic.AuthenticationError:
            return self._error("Clé API refusée : vérifiez-la dans les paramètres")
        except anthropic.PermissionDeniedError:
            return self._error("Cette clé API n'a pas accès au modèle")
        except anthropic.RateLimitError:
            return self._error("Limite de requêtes de l'API atteinte")
        except anthropic.APIStatusError as exc:
            return self._error(f"Erreur API {exc.status_code}")
        except anthropic.APIConnectionError:
            return self._error("Pas de connexion à l'API")
        except Exception as exc:  # clé absente, réponse illisible…
            return self._error(f"{type(exc).__name__}: {exc}"[:160])

        text = next((b.text for b in response.content if b.type == "text"), None)
        if response.stop_reason != "end_turn" or text is None:
            return self._error("Identification refusée ou réponse incomplète")
        try:
            identity = VehicleIdentity.model_validate_json(text)
        except ValueError:
            return self._error("Réponse de l'API illisible")
        self.last_error = None
        result = identity.model_dump()
        result["status"] = "done"
        return result

    def _error(self, message: str) -> dict:
        self.last_error = message
        return {"status": "error", "error": message}

    def status(self, cfg) -> dict:
        with self._lock:
            return {
                "enabled": cfg.vehicle_id_enabled,
                "identified": self.identified,
                "failed": self.failed,
                "skipped": self.skipped,
                "queued": self._queue.qsize(),
                "last_error": self.last_error,
                "top_makes": [{"make": m, "count": c} for m, c in self._makes.most_common(8)],
            }
