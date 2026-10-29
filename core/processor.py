import threading
import time
from typing import Optional

import cv2

from .config import ground_distance_fn
from .detector import draw_visuals
from .riders import RiderDetector
from .vehicle_id import VehicleIdentifier

# Mode auto, sources en direct : sous ce débit, une voiture bouge trop entre deux
# images pour être suivie de façon fiable. On réduit alors la largeur d'analyse.
MIN_LIVE_FPS = 5.0
AUTO_WIDTH_STEPS = (480, 384, 320)
AUTO_SETTLE_FRAMES = 8      # images à mesurer avant de juger une largeur
AUTO_UPGRADE_FRAMES = 150   # stabilité exigée avant de retenter plus grand


class VideoProcessor:
    """Analyse une source une seule fois et partage le JPEG à tous les clients."""

    def __init__(self, stream, detector, counter, config_manager):
        self.stream = stream
        self.detector = detector
        self.counter = counter
        self.config_manager = config_manager
        self.riders = RiderDetector()
        self.identifier = VehicleIdentifier(config_manager, counter)
        self._condition = threading.Condition()
        self._process_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._jpeg_sequence = 0
        self._running = False
        self._thread = None
        self._last_source_sequence = -1
        # Temps de calcul pur par image (détection + comptage + rendu + JPEG), lissé.
        # Contrairement au FPS obtenu, il n'inclut pas l'attente de la source :
        # 1 / ce temps = cadence maximale que l'appareil peut soutenir.
        self.processing_ms: float = 0.0
        # Distance réellement utilisée entre les deux lignes (perspective incluse)
        self.speed_distance_m: float = 0.0
        self.speed_perspective: bool = False
        self.inference_width: int = 0
        self._auto_width: Optional[int] = None
        self._frames_at_width = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="video-processor", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def reset_tracking(self):
        with self._process_lock:
            self.detector.reset_tracker()
            self.riders.reset()
            self.identifier.reset()
            self._last_source_sequence = -1
            # Nouvelle source : performances inconnues, on repart de la largeur configurée.
            self._auto_width = None
            self._frames_at_width = 0
            self.processing_ms = 0.0

    def _loop(self):
        previous_time = time.monotonic()
        metric_time = previous_time
        while self._running:
            cfg = self.config_manager.get()
            interval = 0.0 if cfg.auto_performance else 1.0 / max(1, cfg.target_fps)
            success, frame, sequence, timeline = self.stream.read_frame()
            if not success or frame is None or sequence == self._last_source_sequence:
                time.sleep(0.005)
                continue

            elapsed = time.monotonic() - previous_time
            if elapsed < interval:
                time.sleep(min(0.01, interval - elapsed))
                continue

            self._last_source_sequence = sequence
            started = time.monotonic()
            height, width = frame.shape[:2]
            line_start = (int(cfg.line_start_x * width), int(cfg.line_start_y * height))
            line_end = (int(cfg.line_end_x * width), int(cfg.line_end_y * height))
            speed_line_start = (int(cfg.line_start_x * width), int(cfg.speed_line_y * height))
            speed_line_end = (int(cfg.line_end_x * width), int(cfg.speed_line_y * height))
            ground_distance = ground_distance_fn(cfg, height)
            speed_distance = cfg.speed_distance_m
            if ground_distance is not None:
                speed_distance = abs(ground_distance(line_start[1]) - ground_distance(speed_line_start[1]))
            self.speed_distance_m = speed_distance
            self.speed_perspective = ground_distance is not None

            detect_riders = cfg.detect_riders and "motorcycle" in cfg.selected_classes

            # Une trame défectueuse ne doit jamais arrêter définitivement le thread.
            try:
                if self.stream.source_type == "demo":
                    detections = self.stream.get_demo_detections()
                else:
                    with self._process_lock:
                        self.inference_width = self._select_width(cfg)
                        detections = self.detector.track(
                            frame, cfg.confidence, cfg.iou_threshold,
                            cfg.selected_classes, self.inference_width,
                            include_persons=detect_riders,
                        )
                        if detect_riders:
                            detections = self.riders.process(detections, timeline)

                self.counter.update_tracks(
                    detections, line_start, line_end, cfg.line_buffer_px,
                    timeline=timeline,
                    confirm_frames=cfg.detection_confirm_frames,
                    speed_line_start=speed_line_start,
                    speed_line_end=speed_line_end,
                    speed_distance_m=speed_distance,
                    ground_distance=ground_distance,
                    frame_size=(width, height),
                    merge_fragments=cfg.merge_track_fragments,
                    count_exits=cfg.count_exits,
                )
                self.identifier.observe(frame, detections, timeline, cfg)
            except Exception as exc:
                print(f"[VideoProcessor] Erreur d'analyse: {exc}")
                self.stream.status_message = f"Erreur d'analyse: {type(exc).__name__}"
                self.stream.acknowledge(sequence)
                previous_time = time.monotonic()
                time.sleep(0.05)
                continue
            self.stream.acknowledge(sequence)
            completed = time.monotonic()
            self.counter.fps = 1.0 / max(0.0001, completed - metric_time)
            metric_time = completed
            try:
                tracks, counted_ids, detected_ids = self.counter.get_visual_state()
                annotated = draw_visuals(
                    frame, detections, tracks, counted_ids, detected_ids, line_start, line_end,
                    self.counter.get_stats(), cfg.show_boxes, cfg.show_labels,
                    cfg.show_trails, cfg.show_line, cfg.show_hud,
                    speed_line_start, speed_line_end,
                    speed_distance, cfg.show_speed_line,
                )
                ok, encoded = cv2.imencode(
                    ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality]
                )
            except Exception as exc:
                self.stream.status_message = f"Erreur de traitement: {type(exc).__name__}"
                self.stream.acknowledge(sequence)
                time.sleep(0.05)
                continue
            if ok:
                with self._condition:
                    self._jpeg = encoded.tobytes()
                    self._jpeg_sequence += 1
                    self._condition.notify_all()
            previous_time = time.monotonic()
            work_ms = (previous_time - started) * 1000.0
            self.processing_ms = (
                work_ms if self.processing_ms <= 0 else 0.8 * self.processing_ms + 0.2 * work_ms
            )
            with self._process_lock:
                self._adapt_width(cfg)

    def _width_steps(self, cfg):
        return [cfg.process_width] + [w for w in AUTO_WIDTH_STEPS if w < cfg.process_width]

    def _select_width(self, cfg) -> int:
        """Largeur d'analyse : configurée, ou réduite par le mode auto en direct.

        Les fichiers restent analysés à la largeur configurée : chaque image y est
        traitée sans perte, la lenteur n'y coûte que du temps, pas de la précision.
        """
        fixed = getattr(self.detector, "fixed_imgsz", None)
        if fixed:
            self._auto_width = None
            return fixed
        live = self.stream.source_type in ("webcam", "rtsp", "http", "csi")
        if not (cfg.auto_performance and live):
            self._auto_width = None
            return cfg.process_width
        if self._auto_width not in self._width_steps(cfg):
            self._auto_width = cfg.process_width
            self._frames_at_width = 0
        return self._auto_width

    def _adapt_width(self, cfg):
        if self._auto_width is None:
            return
        self._frames_at_width += 1
        if self._frames_at_width < AUTO_SETTLE_FRAMES:
            return
        steps = self._width_steps(cfg)
        index = steps.index(self._auto_width)
        capacity = self.capacity_fps
        if capacity < MIN_LIVE_FPS and index + 1 < len(steps):
            new_width = steps[index + 1]
        elif (index > 0 and self._frames_at_width >= AUTO_UPGRADE_FRAMES and
              capacity * (self._auto_width / steps[index - 1]) ** 2 >= MIN_LIVE_FPS * 1.3):
            # Coût de YOLO ≈ proportionnel au nombre de pixels ; marge de 30 %
            # pour ne pas osciller entre deux largeurs.
            new_width = steps[index - 1]
        else:
            return
        print(f"[VideoProcessor] Mode auto : {capacity:.1f} FPS max, analyse "
              f"{self._auto_width} -> {new_width} px")
        self._auto_width = new_width
        self._frames_at_width = 0
        self.processing_ms = 0.0

    def wait_for_jpeg(self, after_sequence: int, timeout: float = 2.0):
        with self._condition:
            self._condition.wait_for(
                lambda: not self._running or self._jpeg_sequence > after_sequence,
                timeout=timeout,
            )
            return self._jpeg, self._jpeg_sequence

    def get_jpeg(self) -> Optional[bytes]:
        with self._condition:
            return self._jpeg

    @property
    def capacity_fps(self) -> float:
        return 1000.0 / self.processing_ms if self.processing_ms > 0 else 0.0

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()
