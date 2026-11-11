# app/routes/dashboard.py

import os
import time
import cv2
import psutil
import json
from flask import (
    Blueprint, render_template, Response,
    stream_with_context, current_app, jsonify,
    request, flash, send_file, abort
)

# 🔐 Importar decorador de autenticación
from app.routes.auth import login_required

# -----------------------------------------------------------------------------  
# Blueprint
# -----------------------------------------------------------------------------  

dashboard_bp = Blueprint('dashboard', __name__)

# -----------------------------------------------------------------------------  
# Vistas HTML
# -----------------------------------------------------------------------------  

@dashboard_bp.route('/')
@login_required
def index():
    """Página principal del dashboard (grid de cámaras)"""
    cam_ids = current_app.camera_manager.device_ids
    return render_template('index.html', camera_ids=cam_ids)

# -----------------------------------------------------------------------------  
# Configuración de Notificaciones
# -----------------------------------------------------------------------------  

@dashboard_bp.route("/notifications", methods=["GET", "POST"])
@login_required
def notifications_config():
    config_path = os.path.join("config", "notifications.json")

    # Leer configuración actual
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "email_enabled": False,
            "email_to": "",
            "telegram_enabled": False,
            "telegram_chat_id": "",
            "cooldown_seconds": 60
        }

    if request.method == "POST":
        config["email_enabled"] = "email_enabled" in request.form
        config["telegram_enabled"] = "telegram_enabled" in request.form
        config["email_to"] = request.form.get("email_to", "").strip()
        config["telegram_chat_id"] = request.form.get("telegram_chat_id", "").strip()
        config["cooldown_seconds"] = int(request.form.get("cooldown_seconds", 60))

        try:
            os.makedirs("config", exist_ok=True)
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            flash("✅ Configuración guardada con éxito.", "success")
        except Exception as e:
            flash(f"❌ Error al guardar configuración: {e}", "error")

    return render_template("notifications.html", config=config)

# -----------------------------------------------------------------------------  
# MJPEG Streaming
# -----------------------------------------------------------------------------  

@dashboard_bp.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    """Endpoint MJPEG: /dashboard/video_feed/<camera_id>"""
    return Response(stream_with_context(gen_mjpeg(camera_id)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_mjpeg(camera_id: int):
    """Generador MJPEG usando el CameraManager del servidor."""
    cam = current_app.camera_manager.get_camera(camera_id)
    if cam is None:
        return

    # Si el usuario la apagó, abrir el video no la vuelve a encender
    if not cam.running and not cam.apagada:
        try:
            cam.start()
        except RuntimeError:
            return
        time.sleep(0.1)

    while cam.running:                     # al apagarla, el video termina
        frame = cam.read_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        ok, jpeg = cv2.imencode('.jpg', frame)
        if not ok:
            time.sleep(0.01)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

        fps = getattr(cam, 'fps', 30)
        time.sleep(1.0 / fps)

# -----------------------------------------------------------------------------  
# Snapshot único (prueba rápida)
# -----------------------------------------------------------------------------  

@dashboard_bp.route('/snapshot/<int:camera_id>')
@login_required
def snapshot(camera_id):
    """Devuelve una única imagen JPEG capturada en el momento."""
    cam = current_app.camera_manager.get_camera(camera_id)
    if cam is None:
        abort(404)
    if cam.apagada:
        return "La cámara está apagada", 409
    if not cam.running:
        cam.start()
        time.sleep(0.1)

    frame = cam.read_frame()
    if frame is None:
        return "No hay frame disponible", 503

    ok, jpeg = cv2.imencode('.jpg', frame)
    if not ok:
        return "Error al codificar JPEG", 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# -----------------------------------------------------------------------------  
# Métricas del sistema
# -----------------------------------------------------------------------------  

@dashboard_bp.route('/metrics')
@login_required
def metrics():
    """Devuelve uso de CPU, RAM y disco en JSON."""
    data = {
        'cpu': psutil.cpu_percent(),
        'mem': psutil.virtual_memory().percent,
        'disk': psutil.disk_usage(current_app.config.get('DATA_PATH', '/')).percent,
    }
    return jsonify(data)

# -----------------------------------------------------------------------------  
# Eventos SSE (detecciones IA en tiempo real)
# -----------------------------------------------------------------------------  

@dashboard_bp.route('/events/<int:camera_id>')
@login_required
def stream_events(camera_id):
    """
    Stream de eventos IA (rostros, movimiento, gestos, conteo, cruces) vía SSE.
    Lee del EventStore por id: cada pestaña lleva su propio cursor y ninguna
    le quita eventos a otra (hallazgo 5). Al reconectar, el navegador envía
    Last-Event-ID y se reanuda donde se quedó.
    """
    from app.system_monitor import get_system_metrics
    store = current_app.event_store
    try:
        last_id = int(request.headers.get('Last-Event-ID', 0))
    except ValueError:
        last_id = 0
    if last_id == 0:
        last_id = max(0, store.last_id - 20)   # al abrir, muestra los últimos eventos

    def event_stream(last_id):
        while True:
            try:
                events = store.recent(since_id=last_id, camera_id=camera_id)
                if events:
                    last_id = events[-1]['id']
                payload = list(events) + [get_system_metrics()]
                yield f"id: {last_id}\ndata: {json.dumps(payload, default=str)}\n\n"
            except Exception as e:
                print(f"[ERROR SSE CAM {camera_id}]: {e}")
                yield "data: []\n\n"
            time.sleep(1)

    return Response(stream_with_context(event_stream(last_id)), mimetype='text/event-stream')


# -----------------------------------------------------------------------------
# Fase 2: análisis y gestión de eventos
# -----------------------------------------------------------------------------

@dashboard_bp.route('/eventos')
@login_required
def eventos():
    """Página de consulta, revisión y notas de eventos guardados."""
    cam_ids = current_app.camera_manager.device_ids
    return render_template('eventos.html', camera_ids=cam_ids)


@dashboard_bp.route('/event_image/<int:event_id>')
@login_required
def event_image(event_id):
    """Foto de un evento (solo archivos dentro de sessions/)."""
    store = current_app.event_store
    ev = store.get(event_id)
    path = store.image_file(ev.get('image_path')) if ev else None
    if not path:
        abort(404)
    return send_file(path, mimetype='image/jpeg', max_age=3600)


# -----------------------------------------------------------------------------
# Fase 4: reportes y estadísticas
# -----------------------------------------------------------------------------

@dashboard_bp.route('/reportes')
@login_required
def reportes():
    """Reportes por periodo: actividad, mapa de calor, objetos, personas, cruces; exporta CSV e imprime."""
    cam_ids = current_app.camera_manager.device_ids
    return render_template('reportes.html', camera_ids=cam_ids)


@dashboard_bp.route('/event_file')
@login_required
def event_file():
    """
    Foto de un evento por su ruta (los eventos en vivo aún no tienen el id de la BD).
    Solo sirve archivos dentro de sessions/ (EventStore.image_file valida la ruta).
    """
    path = current_app.event_store.image_file(request.args.get('p', ''))
    if not path:
        abort(404)
    return send_file(path, mimetype='image/jpeg', max_age=3600)
