import json
import os
import threading
from typing import List, Dict, Any, Callable, Optional
from pydantic import BaseModel, Field

# Mapping COCO class IDs to Vehicle names
COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

CLASS_NAME_TO_ID = {v: k for k, v in COCO_VEHICLE_CLASSES.items()}

# French labels for display
CLASS_LABELS_FR = {
    "car": "Voiture",
    "motorcycle": "Moto",
    "bus": "Bus",
    "truck": "Camion"
}


class AppConfig(BaseModel):
    # Video Source
    source_type: str = Field(default="demo", description="webcam, file, rtsp, demo")
    source_path: str = Field(default="0", description="Camera index (0, 1), video file path, or RTSP URL")
    
    # Model & Inference
    model_name: str = Field(default="yolo11n.pt", description="YOLO model name or path")
    confidence: float = Field(default=0.20, ge=0.05, le=1.0, description="YOLO detection confidence threshold")
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0, description="IoU threshold for NMS")

    selected_classes: List[str] = Field(
        default=["car", "motorcycle", "bus", "truck"],
        description="Vehicles to track and count"
    )

    
    # Counting Line Coordinates (normalized 0.0 - 1.0)
    line_start_x: float = Field(default=0.05, ge=0.0, le=1.0)
    line_start_y: float = Field(default=0.55, ge=0.0, le=1.0)
    line_end_x: float = Field(default=0.95, ge=0.0, le=1.0)
    line_end_y: float = Field(default=0.55, ge=0.0, le=1.0)
    line_buffer_px: int = Field(default=20, ge=5, le=100, description="Crossing detection buffer in pixels")
    
    # Display Overlays
    show_boxes: bool = True
    show_labels: bool = True
    show_trails: bool = True
    show_line: bool = True
    show_hud: bool = True
    
    # Performance
    target_fps: int = Field(default=12, ge=1, le=60, description="Cadence maximale d'inférence")
    auto_performance: bool = Field(
        default=True,
        description="Traite les images aussi vite que le matériel le permet",
    )
    process_width: int = Field(default=640, ge=320, le=1920, description="Largeur d'inférence")
    jpeg_quality: int = Field(default=75, ge=40, le=95, description="Qualité du flux MJPEG")
    detection_confirm_frames: int = Field(
        default=2, ge=1, le=10,
        description="Nombre d'images requises avant d'ajouter un véhicule détecté",
    )
    speed_distance_m: float = Field(default=10.0, gt=0.1, le=1000.0)
    speed_line_y: float = Field(default=0.55, ge=0.0, le=1.0)
    show_speed_line: bool = True
    # Compte comme deux-roues une personne qui se déplace à vitesse de véhicule
    detect_riders: bool = True
    # Rattache au même véhicule un nouveau numéro de suivi apparu là où il a été perdu
    merge_track_fragments: bool = True
    # Compte aussi, dans son sens, un véhicule qui disparaît sans avoir franchi la ligne
    count_exits: bool = True
    # Correction de perspective (facultative) : distance au sol en mètres d'une rangée
    # normalisée v ∈ [0, 1], Z(v) = (a·v + b) / (c·v + d). Valable seulement pour la
    # source sur laquelle elle a été mesurée (même caméra, même cadrage).
    speed_ground_map: Optional[List[float]] = Field(default=None, min_length=4, max_length=4)
    speed_ground_map_source: Optional[str] = None
    # Identification marque / modèle par l'API Claude, pour les véhicules comptés.
    # Sans clé saisie, le SDK utilise ANTHROPIC_API_KEY. La clé n'est jamais renvoyée au dashboard.
    vehicle_id_enabled: bool = False
    vehicle_id_api_key: str = ""
    vehicle_id_model: str = "claude-opus-5-5"
    vehicle_id_max_per_minute: int = Field(default=20, ge=1, le=120)


def ground_distance_fn(cfg: AppConfig, frame_height: int) -> Optional[Callable[[float], float]]:
    """Convertit une rangée d'image (px) en distance au sol (m), ou None sans calibration."""
    if not cfg.speed_ground_map or cfg.speed_ground_map_source != cfg.source_path:
        return None
    a, b, c, d = cfg.speed_ground_map

    def distance(y_px: float) -> float:
        v = y_px / frame_height
        return (a * v + b) / (c * v + d)

    return distance


class ConfigManager:
    """Thread-safe configuration manager with JSON persistence."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._lock = threading.Lock()
        self._config = AppConfig()
        self.load()

    def get(self) -> AppConfig:
        with self._lock:
            return self._config.model_copy()

    def update(self, new_data: Dict[str, Any]) -> AppConfig:
        with self._lock:
            current_dict = self._config.model_dump()
            current_dict.update(new_data)
            self._config = AppConfig(**current_dict)
            self.save()
            return self._config.model_copy()

    def save(self):
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self._config.model_dump(), f, indent=2)
            # Peut contenir des identifiants RTSP : lisible par le seul propriétaire.
            os.chmod(self.config_file, 0o600)
        except Exception as e:
            print(f"[ConfigManager] Erreur sauvegarde config: {e}")

    def load(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._config = AppConfig(**data)
            except Exception as e:
                print(f"[ConfigManager] Erreur chargement config: {e}")
