import os
os.environ["MPLCONFIGDIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
import io
import csv
import re

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
import uvicorn

from core.config import ConfigManager, CLASS_LABELS_FR
from core.detector import YOLODetector
from core.counter import VehicleCounter, UNCOUNTED_REASONS
from core.video_stream import VideoStreamManager, opencv_has_gstreamer
from core.processor import VideoProcessor


# Répertoires
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Gestionnaires globaux
config_mgr = ConfigManager(config_file=os.path.join(BASE_DIR, "config.json"))
detector = YOLODetector(model_name=config_mgr.get().model_name)
counter = VehicleCounter()

cfg = config_mgr.get()
video_stream = VideoStreamManager(source_type=cfg.source_type, source_path=cfg.source_path)
processor = VideoProcessor(video_stream, detector, counter, config_mgr)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    processor.start()
    try:
        yield
    finally:
        processor.stop()
        video_stream.release()


app = FastAPI(
    title="KreyatiSky TrafficVision",
    description="Système intelligent et léger de détection et comptage de véhicules avec YOLO",
    version="1.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def validate_source(stype, spath):
    """Normalise et valide une source vidéo; lève ValueError si elle est refusée."""
    stype = str(stype).lower()
    # Un nom de fichier peut légitimement contenir des espaces en bordure.
    spath = str(spath) if stype == "file" else str(spath).strip()
    if stype not in {"demo", "webcam", "csi", "file", "rtsp", "http"}:
        raise ValueError("Type de source non pris en charge")
    if stype in {"webcam", "csi"} and (not spath.isdigit() or int(spath) < 0):
        raise ValueError("L'index de la caméra doit être un entier positif ou nul (0, 1, 2...)")
    if stype in {"rtsp", "http"} and not re.match(r"^(rtsp|rtsps|http|https)://", spath, re.I):
        raise ValueError("URL réseau invalide")
    if stype == "file":
        # Le dashboard ne connaît que le nom du fichier, relatif au dossier uploads.
        candidate = (Path(UPLOADS_DIR) / spath).resolve()
        if Path(UPLOADS_DIR).resolve() not in candidate.parents or not candidate.is_file():
            raise ValueError("Le fichier doit provenir du dossier uploads")
        spath = str(candidate)
    return stype, spath


def public_source_path(stype: str, spath: str) -> str:
    """Version affichable : sans mot de passe réseau ni chemin absolu du serveur."""
    if stype == "file":
        return os.path.basename(spath)
    if stype in {"rtsp", "http"}:
        parts = urlsplit(spath)
        if "@" in parts.netloc:
            userinfo, host = parts.netloc.rsplit("@", 1)
            if ":" in userinfo:
                user = userinfo.split(":", 1)[0]
                return urlunsplit(parts._replace(netloc=f"{user}:***@{host}"))
    return spath


def restore_hidden_source(stype, spath, current) -> str:
    """Le dashboard renvoie la version masquée de la source actuelle : on rétablit la vraie."""
    public = public_source_path(current.source_type, current.source_path)
    if str(stype).lower() == current.source_type and str(spath).strip() == public.strip():
        return current.source_path
    return spath


def public_config(cfg) -> dict:
    data = cfg.model_dump()
    data["source_path"] = public_source_path(cfg.source_type, cfg.source_path)
    # Chemin serveur interne ; la calibration n'est pas modifiable depuis le dashboard.
    data.pop("speed_ground_map_source", None)
    data["vehicle_id_key_set"] = bool(data.pop("vehicle_id_api_key", ""))
    return data


def switch_source(stype: str, spath: str, replay: bool = False):
    """Ouvre la nouvelle source et repart de zéro.

    Bloquant (jusqu'à ~2,5 s pour un flux réseau, attente de l'inférence en cours) :
    les routes l'appellent via run_in_threadpool pour ne pas figer la boucle asynchrone.
    """
    video_stream.change_source(stype, spath)
    if replay:
        video_stream.replay_video()
    counter.reset()
    processor.reset_tracking()


def restart_video():
    video_stream.replay_video()
    counter.reset()
    processor.reset_tracking()


def generate_frames():
    """Diffuse le résultat partagé; aucun calcul YOLO n'est fait par client."""
    sequence = -1
    # Sans traitement actif, wait_for_jpeg rend la main immédiatement : on termine
    # le flux au lieu de boucler à vide.
    while processor.is_running:
        frame_bytes, new_sequence = processor.wait_for_jpeg(sequence)
        if frame_bytes is None or new_sequence == sequence:
            continue
        sequence = new_sequence
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Page principale du tableau de bord."""
    cfg = config_mgr.get()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "config": cfg,
        "class_labels": CLASS_LABELS_FR,
        "source_status": video_stream.status_message
    })


@app.get("/video_feed")
async def video_feed():
    """Flux vidéo MJPEG en direct pour la balise <img> du dashboard."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/stats")
async def get_stats():
    """Statistiques en temps réel pour le rafraîchissement JS du dashboard."""
    stats = counter.get_stats()
    stats["source_status"] = video_stream.status_message
    stats["source_type"] = video_stream.source_type
    stats["is_connected"] = video_stream.is_connected
    stats["source_health"] = video_stream.get_health()
    stats["video_progress"] = video_stream.get_video_progress()
    settings = config_mgr.get()
    stats["speed_calibration_distance_m"] = round(processor.speed_distance_m or settings.speed_distance_m, 1)
    stats["speed_perspective"] = processor.speed_perspective
    stats["processor_running"] = processor.is_running
    stats["capacity_fps"] = round(processor.capacity_fps, 1)
    stats["processing_ms"] = round(processor.processing_ms)
    stats["inference_width"] = processor.inference_width
    stats["configured_width"] = settings.process_width
    stats["auto_performance"] = settings.auto_performance
    stats["vehicle_id"] = processor.identifier.status(settings)
    stats["compute_device"] = detector.device
    return JSONResponse(stats)


@app.post("/api/video/control")
async def video_control(request: Request):
    """Contrôle de la lecture vidéo : pause, replay."""
    try:
        data = await request.json()
        action = data.get("action", "pause")
        
        if action == "replay":
            await run_in_threadpool(restart_video)
            return JSONResponse({"status": "success", "message": "Vidéo recommencée", "progress": video_stream.get_video_progress()})
        elif action == "pause":
            is_paused = video_stream.toggle_pause()
            return JSONResponse({"status": "success", "is_paused": is_paused, "progress": video_stream.get_video_progress()})
        else:
            return JSONResponse({"status": "error", "message": "Action inconnue"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)


@app.get("/api/settings")
async def get_settings():
    """Récupère les paramètres actuels."""
    return JSONResponse(public_config(config_mgr.get()))


@app.post("/api/settings")
async def update_settings(request: Request):
    """Mise à jour des paramètres à la volée."""
    try:
        data = await request.json()
        current_cfg = config_mgr.get()
        old_stype = current_cfg.source_type
        old_spath = current_cfg.source_path

        # Un fichier .pt est chargé via pickle (exécution de code possible) : le modèle
        # ne se change que dans config.json, suivi d'un redémarrage.
        if "model_name" in data and data["model_name"] != current_cfg.model_name:
            raise ValueError("Le modèle YOLO se change dans config.json puis en redémarrant le serveur")
        # Champ vide = clé inchangée ; la clé enregistrée n'est jamais renvoyée au navigateur.
        if "vehicle_id_api_key" in data:
            key = str(data["vehicle_id_api_key"] or "").strip()
            if key:
                data["vehicle_id_api_key"] = key
            else:
                data.pop("vehicle_id_api_key")
        data.pop("vehicle_id_key_set", None)

        # Même validation que /api/source : ce point d'entrée ne doit pas la contourner.
        if "source_type" in data or "source_path" in data:
            stype = data.get("source_type", old_stype)
            spath = restore_hidden_source(stype, data.get("source_path", old_spath), current_cfg)
            if str(stype).lower() != old_stype or spath != old_spath:
                data["source_type"], data["source_path"] = validate_source(stype, spath)
            else:
                data["source_type"], data["source_path"] = old_stype, old_spath

        updated_cfg = config_mgr.update(data)
        
        new_stype = updated_cfg.source_type
        new_spath = updated_cfg.source_path

        if new_stype != old_stype or new_spath != old_spath:
            await run_in_threadpool(switch_source, new_stype, new_spath)

        return JSONResponse({"status": "success", "config": public_config(updated_cfg)})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)



@app.post("/api/reset")
async def reset_counts():
    """Réinitialise les compteurs."""
    counter.reset()
    processor.identifier.reset()
    return JSONResponse({"status": "success", "message": "Compteurs réinitialisés"})


@app.post("/api/source")
async def change_source(request: Request):
    """Change la source vidéo (Webcam, RTSP, Fichier, Démo)."""
    try:
        data = await request.json()
        stype = data.get("source_type", "demo")
        spath = restore_hidden_source(stype, data.get("source_path", "0"), config_mgr.get())
        stype, spath = validate_source(stype, spath)

        config_mgr.update({"source_type": stype, "source_path": spath})
        await run_in_threadpool(switch_source, stype, spath)

        if not video_stream.is_connected:
            # Une caméra en direct peut se connecter plus tard ; une caméra CSI sans
            # GStreamer, jamais.
            pending = stype in {"webcam", "rtsp", "http"} or (stype == "csi" and opencv_has_gstreamer())
            return JSONResponse({
                "status": "pending" if pending else "error",
                "message": video_stream.status_message,
                "source_status": video_stream.status_message,
                "health": video_stream.get_health(),
            }, status_code=202 if pending else 422)

        return JSONResponse({
            "status": "success",
            "message": f"Source changée: {stype} ({public_source_path(stype, spath)})",
            "source_status": video_stream.status_message,
            "health": video_stream.get_health(),
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)



@app.get("/api/uploaded_videos")
async def list_uploaded_videos():
    """Liste les vidéos déjà téléversées sur le serveur."""
    videos = []
    if os.path.exists(UPLOADS_DIR):
        for fname in os.listdir(UPLOADS_DIR):
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.m4v')):
                full_p = os.path.join(UPLOADS_DIR, fname)
                videos.append({
                    "name": fname,
                    "path": fname,
                    "size_mb": round(os.path.getsize(full_p) / (1024 * 1024), 2)
                })
    return JSONResponse({"videos": videos})


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Permet à l'utilisateur de téléverser un fichier vidéo à analyser."""
    try:
        allowed = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}
        original_name = Path(file.filename or "video").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in allowed:
            return JSONResponse({"status": "error", "message": "Format vidéo non autorisé"}, status_code=400)
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem).strip("._") or "video"
        # Ne jamais écraser une vidéo existante (peut-être en cours de lecture).
        file_path = os.path.join(UPLOADS_DIR, f"{safe_stem}{suffix}")
        index = 1
        while os.path.exists(file_path):
            file_path = os.path.join(UPLOADS_DIR, f"{safe_stem}_{index}{suffix}")
            index += 1
        max_bytes = 2 * 1024 * 1024 * 1024
        written = 0
        with open(file_path, "xb") as f:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    os.unlink(file_path)
                    return JSONResponse({"status": "error", "message": "Vidéo trop volumineuse (2 Go max)"}, status_code=413)
                f.write(chunk)
        
        # Basculer immédiatement vers la vidéo uploadée
        config_mgr.update({"source_type": "file", "source_path": file_path})
        await run_in_threadpool(switch_source, "file", file_path, True)

        if not video_stream.is_connected:
            return JSONResponse({
                "status": "error",
                "message": "Le fichier a été reçu, mais aucune image vidéo valide n'a pu être décodée.",
                "file_path": os.path.basename(file_path),
                "source_status": video_stream.status_message,
            }, status_code=422)
        
        return JSONResponse({
            "status": "success",
            "filename": os.path.basename(file_path),
            "file_path": os.path.basename(file_path),
            "source_status": video_stream.status_message
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)



@app.get("/api/export/csv")
async def export_csv():
    """Téléchargement de l'historique des passages au format CSV."""
    stats = counter.get_stats()
    events = counter.get_event_log()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["ID Véhicule", "Catégorie", "Statut", "Direction passage", "Sens estimé", "Vitesse max (km/h)", "Raison non compté", "Marque", "Modèle", "Couleur", "Fiabilité identification", "Heure", "Timestamp UNIX"])
    for ev in events:
        identity = ev.get("identity") or {}
        done = identity.get("status") == "done"
        writer.writerow([
            ev.get("id"),
            CLASS_LABELS_FR.get(ev.get("class"), ev.get("class")),
            ev.get("status", ""),
            ("MONTE (UP)" if ev.get("direction") == "up" else
             "DESCEND (DOWN)" if ev.get("direction") == "down" else "EN ATTENTE")
            + (" - sorti sans franchir la ligne" if ev.get("counted_by") == "exit" else ""),
            "MONTE (UP)" if ev.get("motion_direction") == "up" else
            "DESCEND (DOWN)" if ev.get("motion_direction") == "down" else "INCONNU",
            ev.get("max_speed_kmh", ""),
            UNCOUNTED_REASONS.get(ev.get("uncounted_reason"), ""),
            identity.get("make", "") if done else "",
            identity.get("model", "") if done else "",
            identity.get("color", "") if done else "",
            identity.get("confidence", "") if done else "",
            ev.get("time_str"),
            ev.get("timestamp")
        ])
    
    writer.writerow([])
    writer.writerow(["--- RESUME DES COMPTAGES ---"])
    writer.writerow(["Total Montant (UP)", stats.get("count_up")])
    writer.writerow(["Total Descendant (DOWN)", stats.get("count_down")])
    writer.writerow(["Total véhicules détectés", stats.get("total")])
    writer.writerow(["Direction en attente", stats.get("count_pending")])
    writer.writerow(["Vitesse maximale (km/h)", stats.get("max_speed_kmh")])
    writer.writerow(["Vitesse moyenne (km/h)", stats.get("avg_speed_kmh")])
    for item in stats.get("uncounted_reasons", []):
        writer.writerow([f"Non compté : {item['label']}", item["count"]])
    for item in processor.identifier.status(config_mgr.get())["top_makes"]:
        writer.writerow([f"Marque : {item['make']}", item["count"]])
    writer.writerow(["Date Export", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    
    csv_bytes = output.getvalue().encode("utf-8-sig")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )


@app.get("/api/snapshot")
async def snapshot():
    """Capture d'un instantané de l'image courante."""
    jpeg = processor.get_jpeg()
    if jpeg is None:
        return JSONResponse({"status": "error", "message": "Aucune image disponible"}, status_code=500)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"}
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

