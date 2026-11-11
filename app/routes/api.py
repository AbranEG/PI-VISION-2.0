# app/routes/api.py — API REST v1 (Fase 1: eventos persistentes)
from datetime import datetime, timedelta
from flask import Blueprint, jsonify, current_app, request, session

api_bp = Blueprint('api', __name__, url_prefix='/api/v1')


@api_bp.before_request
def _require_login():
    # Antes la API no pedía sesión: cualquiera con la URL de ngrok leía los eventos.
    if 'user_id' not in session:
        return jsonify({"error": "no autenticado"}), 401


def _int(name, default=None):
    v = request.args.get(name)
    try:
        return int(v) if v not in (None, '') else default
    except ValueError:
        return default


def _filters():
    tipo = request.args.get('type')
    return dict(
        type=tipo.split(',') if tipo else None,
        camera_id=_int('camera'),
        label=request.args.get('label') or None,
        since=request.args.get('since') or None,
        until=request.args.get('until') or None,
        reviewed=_bool('reviewed'),
    )


def _bool(name):
    v = request.args.get(name)
    if v in (None, ''):
        return None
    return v.lower() in ('1', 'true', 'si', 'sí', 'yes')


@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """Cámaras configuradas y si están capturando."""
    mgr = current_app.camera_manager
    out = []
    for cam_id in mgr.device_ids:
        cam = mgr.get_camera(cam_id)
        out.append({"id": cam_id, "active": bool(getattr(cam, 'running', False)),
                    "apagada": bool(getattr(cam, 'apagada', False))})
    return jsonify(out), 200


@api_bp.route('/cameras/<int:cam_id>/power', methods=['POST'])
def camera_power(cam_id):
    """
    Enciende o apaga una cámara desde el panel: {"on": true|false}.
    Apagada no captura, no corre la IA y no guarda eventos ni fotos (ahorra disco y CPU).
    """
    cam = current_app.camera_manager.get_camera(cam_id)
    if cam is None:
        return jsonify({"error": f"no existe la cámara {cam_id}"}), 404
    on = (request.get_json(silent=True) or {}).get('on')
    if not isinstance(on, bool):
        return jsonify({"error": "falta 'on' (true o false)"}), 400
    usuario = session.get('username', '?')
    if on:
        try:
            cam.encender()
        except Exception as e:
            current_app.logger.warning(f"No se pudo encender la cámara {cam_id}: {e}")
            return jsonify({"error": "la cámara no respondió; revisa que esté conectada",
                            "id": cam_id, "activa": False, "apagada": True}), 503
    else:
        cam.apagar()
    current_app.logger.info(f"Cámara {cam_id} {'encendida' if on else 'apagada'} por {usuario}")
    return jsonify({"id": cam_id, "activa": bool(cam.running), "apagada": bool(cam.apagada)}), 200


@api_bp.route('/events', methods=['GET'])
def get_events():
    """
    Eventos guardados, del más reciente al más antiguo.
    Filtros: ?type=face,gesture &camera=0 &label=Desconocido
             &since=2026-11-01 &until=2026-11-02 &limit=100 &offset=0
    """
    store = current_app.event_store
    f = _filters()
    limit = min(_int('limit', 100), 1000)
    offset = _int('offset', 0)
    items = store.query(limit=limit, offset=offset, **f)
    return jsonify({"total": store.count(**f), "limit": limit, "offset": offset, "items": items}), 200


@api_bp.route('/events/live', methods=['GET'])
def get_live_events():
    """Eventos en memoria con id > since_id (tiempo real, sin borrar nada)."""
    store = current_app.event_store
    items = store.recent(since_id=_int('since_id', 0), camera_id=_int('camera'), limit=_int('limit', 200))
    return jsonify({"last_id": store.last_id, "items": items}), 200


@api_bp.route('/events/summary', methods=['GET'])
def get_summary():
    """Conteos por tipo, cámara, hora y etiqueta. Por defecto, las últimas 24 h."""
    since = request.args.get('since') or (datetime.now() - timedelta(hours=24)).isoformat(sep=' ')
    until = request.args.get('until') or None
    return jsonify(current_app.event_store.summary(since=since, until=until, camera_id=_int('camera'))), 200


@api_bp.route('/events/stats', methods=['GET'])
def get_store_stats():
    """Estado de la capa de persistencia: encolados, guardados, perdidos, búfer."""
    return jsonify(current_app.event_store.stats()), 200


# ----------------------------------------------------------------------------- Fase 2: gestión
@api_bp.route('/events/<int:event_id>', methods=['GET'])
def get_event(event_id):
    ev = current_app.event_store.get(event_id)
    if not ev:
        return jsonify({"error": "no existe"}), 404
    return jsonify(ev), 200


@api_bp.route('/events/<int:event_id>', methods=['PATCH'])
def update_event(event_id):
    """Cuerpo JSON: {"reviewed": true|false, "notes": "texto"} (cualquiera de los dos)."""
    body = request.get_json(silent=True) or {}
    reviewed = body.get('reviewed')
    notes = body.get('notes')
    if reviewed is None and notes is None:
        return jsonify({"error": "nada que actualizar"}), 400
    store = current_app.event_store
    if not store.update(event_id, reviewed=reviewed, notes=notes):
        return jsonify({"error": "no existe"}), 404
    return jsonify(store.get(event_id)), 200


@api_bp.route('/events/review', methods=['POST'])
def bulk_review():
    """Cuerpo JSON: {"ids": [1, 2, 3], "reviewed": true}"""
    body = request.get_json(silent=True) or {}
    ids = body.get('ids') or []
    if not isinstance(ids, list):
        return jsonify({"error": "ids debe ser una lista"}), 400
    n = current_app.event_store.bulk_review(ids, reviewed=body.get('reviewed', True))
    return jsonify({"actualizados": n}), 200


# ----------------------------------------------------------------------------- Fase 4: reportes
@api_bp.route('/reports', methods=['GET'])
def get_report():
    """
    Reporte agregado. ?since=2026-11-01&until=2026-11-08&camera=0&bucket=hour|day
    Por defecto: últimos 7 días.
    """
    since = request.args.get('since') or (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d %H:%M')
    until = request.args.get('until') or None
    bucket = request.args.get('bucket') or None
    return jsonify(current_app.event_store.report(since, until, camera_id=_int('camera'), bucket=bucket)), 200


def _csv_response(header, rows, nombre):
    """CSV con BOM (Excel respeta los acentos) como descarga."""
    import csv
    import io
    from flask import Response
    buf = io.StringIO()
    buf.write('\ufeff')
    wr = csv.writer(buf)
    wr.writerow(header)
    wr.writerows(rows)
    return Response(buf.getvalue(), mimetype='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="{nombre}"'})


@api_bp.route('/events/export.csv', methods=['GET'])
def export_csv():
    """
    Exportación CLARA para usuario final: columnas en español, fecha y hora separadas,
    tipos traducidos, descripción en lenguaje natural y sin JSON.
    Mismos filtros que /events. Máximo 50,000 filas.
    """
    from app.export_format import COLUMNAS, fila_clara
    rows = current_app.event_store.export_rows(**_filters())
    filas = []
    for r in rows:
        f = fila_clara(r)
        filas.append([f[c] for c in COLUMNAS])
    return _csv_response(COLUMNAS, filas, f"pivision_eventos_{datetime.now():%Y%m%d_%H%M}.csv")


@api_bp.route('/events/export_full.csv', methods=['GET'])
def export_full_csv():
    """
    Exportación COMPLETA (técnica): campos originales de la base y el JSON de `datos`.
    Mismos filtros que /events. Máximo 50,000 filas.
    """
    import json as _json
    rows = current_app.event_store.export_rows(**_filters())
    header = ['id', 'fecha', 'camara', 'tipo', 'etiqueta', 'valor', 'revisado', 'notas',
              'foto', 'origen', 'datos']
    filas = [[r['id'], r['ts'], r['camera_id'], r['type'], r['label'] or '',
              r['value'] if r['value'] is not None else '',
              'sí' if r['reviewed'] else 'no', r['notes'] or '', r['image_path'] or '', r['source'],
              _json.dumps(r['data'], ensure_ascii=False)] for r in rows]
    return _csv_response(header, filas, f"pivision_eventos_completo_{datetime.now():%Y%m%d_%H%M}.csv")


# ----------------------------------------------------------------------------- estado del sistema
_inicio = datetime.now()


@api_bp.route('/status', methods=['GET'])
def get_status():
    """Estado para la barra superior y el panel en vivo: cámaras, Hailo, recursos y eventos."""
    from app.system_monitor import get_system_metrics
    mgr = current_app.camera_manager
    camaras, disp, pose = [], None, None
    for cam_id in mgr.device_ids:
        cam = mgr.get_camera(cam_id)
        pipe = getattr(cam, 'pipeline', None)
        obj = getattr(pipe, 'objects', None)
        a = getattr(obj, 'available', None)
        if a is not None:
            disp = bool(disp) or a
        pd = getattr(pipe, '_pose_disponible', None)
        if pd is not None:
            pose = bool(pose) or pd
        camaras.append({"id": cam_id, "activa": bool(getattr(cam, 'running', False)),
                        "apagada": bool(getattr(cam, 'apagada', False)),
                        **{k: v for k, v in (getattr(cam, 'stats', {}) or {}).items()},
                        "hailo": {k: v for k, v in (getattr(obj, '_ultimo_tiempo', {}) or {}).items()}})
    return jsonify({
        "hora": datetime.now().isoformat(timespec='seconds'),
        "en_linea_s": int((datetime.now() - _inicio).total_seconds()),
        "camaras": camaras,
        "hailo": {"disponible": disp, "pose": bool(pose)},
        "sistema": get_system_metrics(),
        "eventos": current_app.event_store.stats(),
    }), 200
