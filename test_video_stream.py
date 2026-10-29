"""Tests légers des sources vidéo, sans caméra ni serveur RTSP obligatoires."""

import os
import tempfile
import time

import cv2
import numpy as np

from core.video_stream import VideoStreamManager


class FakeCapture:
    def __init__(self, frames, opened=True, fps=25.0):
        self.frames = [frame.copy() for frame in frames]
        self.opened = opened
        self.fps = fps
        self.position = 0

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.opened or self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.frames[0].shape[1] if self.frames else 0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.frames[0].shape[0] if self.frames else 0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.position = int(value)
        return True

    def release(self):
        self.opened = False


def manager(source_type, source_path="0"):
    return VideoStreamManager(source_type, source_path, autostart=False)


def test_demo_source():
    stream = manager("demo")
    stream._init_source()
    assert stream.is_connected
    assert stream.get_health()["source_type"] == "demo"


def test_webcam_source_uses_numeric_index():
    stream = manager("webcam", "2")
    seen = []
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    stream._open_capture_with_timeout = lambda source, timeout_sec: seen.append(source) or FakeCapture([frame])
    stream._init_source()
    assert seen == [2]
    assert stream.is_connected and stream.read()[0]
    assert stream.get_health()["width"] == 64


def test_rtsp_and_http_sources():
    frame = np.zeros((36, 52, 3), dtype=np.uint8)
    for source_type, url in (("rtsp", "rtsp://camera/stream"), ("http", "https://camera/video")):
        stream = manager(source_type, url)
        seen = []
        stream._open_capture_with_timeout = lambda source, timeout_sec: seen.append(source) or FakeCapture([frame])
        stream._init_source()
        assert seen == [url]
        assert stream.is_connected
        assert stream.status_message == "Flux réseau connecté"


def test_file_source_and_invalid_stream():
    frame = np.zeros((72, 128, 3), dtype=np.uint8)
    stream = manager("file")
    original_capture = cv2.VideoCapture
    with tempfile.NamedTemporaryFile(suffix=".mp4") as video_file:
        stream.source_path = video_file.name
        cv2.VideoCapture = lambda _path: FakeCapture([frame, frame], fps=20.0)
        try:
            stream._init_source()
        finally:
            cv2.VideoCapture = original_capture
    assert stream.is_connected
    assert stream.total_frames == 2
    assert stream.fps_source == 20.0

    broken = manager("rtsp", "rtsp://offline/stream")
    broken.current_frame = frame
    broken._open_capture_with_timeout = lambda source, timeout_sec: FakeCapture([], opened=True)
    broken._init_source()
    assert not broken.is_connected
    assert broken.cap is None
    assert broken.last_error


def test_stale_reconnect_does_not_reopen_new_source():
    import threading
    stream = manager("rtsp", "rtsp://offline/stream")
    stream.reconnect_attempts = 5  # attente de reconnexion longue (6 s)
    opened = []
    stream._init_source = lambda: opened.append(stream.source_type)
    waiter = threading.Thread(target=stream._reconnect_after_delay)
    waiter.start()
    time.sleep(0.1)
    with stream.lock:  # équivalent de change_source sans ouvrir de vraie capture
        stream.source_type = "demo"
        stream.source_generation += 1
    stream._source_changed.set()
    waiter.join(timeout=1.0)
    assert not waiter.is_alive(), "le changement de source doit interrompre l'attente"
    assert opened == [], "la reconnexion périmée ne doit pas réinitialiser la nouvelle source"


def test_auto_mode_adapts_width_for_live_sources_only():
    from core.config import AppConfig
    from core.processor import VideoProcessor, AUTO_SETTLE_FRAMES, AUTO_UPGRADE_FRAMES
    stream = manager("rtsp", "rtsp://camera/stream")
    proc = VideoProcessor(stream, None, None, None)
    cfg = AppConfig(auto_performance=True, process_width=640)

    def run_frames(count, processing_ms):
        for _ in range(count):
            proc.inference_width = proc._select_width(cfg)
            proc.processing_ms = processing_ms
            proc._adapt_width(cfg)

    run_frames(AUTO_SETTLE_FRAMES, 800)  # 1,25 FPS : trop lent pour le suivi
    assert proc._select_width(cfg) == 480
    run_frames(AUTO_SETTLE_FRAMES * 2, 800)  # 480 -> 384 -> 320
    assert proc._select_width(cfg) == 320
    proc._frames_at_width = 0
    run_frames(AUTO_SETTLE_FRAMES, 800)
    assert proc._select_width(cfg) == 320, "320 px est le plancher"
    proc._frames_at_width = 0
    run_frames(AUTO_UPGRADE_FRAMES - 1, 50)
    assert proc._select_width(cfg) == 320, "pas de remontée avant une longue stabilité"
    run_frames(1, 50)  # 20 FPS stables : on retente plus grand
    assert proc._select_width(cfg) == 384

    stream.source_type = "file"  # un fichier garde toujours la largeur configurée
    assert proc._select_width(cfg) == 640
    cfg_manual = AppConfig(auto_performance=False, process_width=640)
    stream.source_type = "webcam"
    assert proc._select_width(cfg_manual) == 640


def test_tracker_reset_keeps_ultralytics_trackers_usable():
    from types import SimpleNamespace
    from core.detector import YOLODetector

    class FakeTracker:
        resets = 0
        def reset(self):
            self.resets += 1

    tracker = FakeTracker()
    detector = YOLODetector.__new__(YOLODetector)  # sans charger le modèle
    detector.model = SimpleNamespace(predictor=SimpleNamespace(trackers=[tracker]))
    detector.reset_tracker()
    # Mettre trackers à None bloquait toute détection après « Rejouer »
    assert detector.model.predictor.trackers == [tracker]
    assert tracker.resets == 1
    detector.model = SimpleNamespace(predictor=None)  # pas encore d'inférence
    detector.reset_tracker()


def test_live_source_survives_brief_read_failures_and_timestamps_capture():
    import threading
    frame = np.zeros((36, 52, 3), dtype=np.uint8)

    class FlakyCapture(FakeCapture):
        calls = 0
        def read(self):
            FlakyCapture.calls += 1
            if FlakyCapture.calls in (2, 3):     # deux images perdues
                return False, None
            time.sleep(0.01)
            return True, frame.copy()

    stream = manager("rtsp", "rtsp://camera/stream")
    stream._open_capture_with_timeout = lambda source, timeout_sec: FlakyCapture([frame])
    reconnects = []
    stream._reconnect_after_delay = lambda: reconnects.append(1)
    stream._init_source()
    stream.is_running = True
    thread = threading.Thread(target=stream._capture_loop, daemon=True)
    thread.start()
    time.sleep(0.3)
    ok, _, sequence, captured_at = stream.read_frame()
    stream.is_running = False
    thread.join(timeout=1)
    assert reconnects == []                          # pas de coupure pour deux images perdues
    assert ok and sequence > 5
    assert 0 <= time.monotonic() - captured_at < 0.2   # instant de capture, pas de lecture


def test_tensorrt_engine_size_is_fixed():
    import json as _json
    from types import SimpleNamespace
    from core.config import AppConfig
    from core.detector import engine_imgsz
    from core.processor import VideoProcessor

    assert engine_imgsz("yolo11n.pt") is None
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "yolo11n.engine")
        meta = _json.dumps({"imgsz": [480, 480], "task": "detect"}).encode()
        with open(path, "wb") as f:   # en-tête écrit par ultralytics à l'export
            f.write(len(meta).to_bytes(4, byteorder="little") + meta + b"\x00" * 16)
        assert engine_imgsz(path) == 480

    processor = VideoProcessor.__new__(VideoProcessor)
    processor.stream = SimpleNamespace(source_type="csi")
    processor.detector = SimpleNamespace(fixed_imgsz=480)
    processor._auto_width, processor._frames_at_width = None, 0
    assert processor._select_width(AppConfig(process_width=640, auto_performance=True)) == 480


def test_csi_camera_source():
    from core.video_stream import csi_pipeline, opencv_has_gstreamer
    pipeline = csi_pipeline(1)
    assert pipeline.startswith("nvarguscamerasrc sensor-id=1 ")
    assert "appsink drop=true max-buffers=1" in pipeline

    frame = np.zeros((36, 52, 3), dtype=np.uint8)
    stream = manager("csi", "1")
    opened = []
    stream._open_capture_with_timeout = lambda source, timeout_sec, api=0: opened.append((source, api)) or FakeCapture([frame])
    stream._init_source()
    if opencv_has_gstreamer():
        assert opened == [(pipeline, cv2.CAP_GSTREAMER)] and stream.is_connected
    else:   # ce Mac : message clair, aucune tentative d'ouverture
        assert opened == [] and not stream.is_connected
        assert "GStreamer" in stream.status_message

if __name__ == "__main__":
    tests = [
        test_demo_source,
        test_webcam_source_uses_numeric_index,
        test_rtsp_and_http_sources,
        test_file_source_and_invalid_stream,
        test_stale_reconnect_does_not_reopen_new_source,
        test_auto_mode_adapts_width_for_live_sources_only,
        test_tracker_reset_keeps_ultralytics_trackers_usable,
        test_live_source_survives_brief_read_failures_and_timestamps_capture,
        test_tensorrt_engine_size_is_fixed,
        test_csi_camera_source,
    ]
    for test in tests:
        test()
        print(f"OK {test.__name__}")
    print(f"{len(tests)} tests source vidéo réussis")
