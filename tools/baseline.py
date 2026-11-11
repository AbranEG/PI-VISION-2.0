#!/usr/bin/env python3
"""
baseline.py — Medición de rendimiento de PiVision IA (antes / después).

Ejecuta el pipeline real del proyecto (clase Camera + AIPipeline) sin Flask,
envuelve cada etapa con cronómetros y registra FPS, latencia por etapa,
CPU, RAM, temperatura, throttling y eventos guardados en la BD.

No modifica el código medido: instrumenta las clases en memoria.
Las notificaciones (correo/Telegram) se desactivan durante la medición.

Uso:
  python tools/baseline.py --code-dir ~/PiVision-baseline --cams 0 --label cpu_1cam
  python tools/baseline.py --code-dir ~/PiVision-baseline --cams 0 2 --label cpu_2cam
  python tools/baseline.py --code-dir ~/PiVision-baseline --cams 0 --use-tpu --label tpu_1cam

Resultados: <code-dir>/results/baseline_<label>_<fecha>.json y .md
"""
import argparse
import contextlib
import datetime as dt
import functools
import hashlib
import importlib
import importlib.metadata as md
import io
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser(description="Medición de rendimiento PiVision IA")
p.add_argument("--code-dir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
               help="Carpeta del código a medir (por defecto, la que contiene tools/)")
p.add_argument("--cams", nargs="+", default=["0"],
               help="IDs de cámara (0 2 …) o rutas de video para pruebas")
p.add_argument("--duration", type=int, default=60, help="Segundos de medición (sin contar warmup)")
p.add_argument("--warmup", type=int, default=10, help="Segundos iniciales que se descartan")
p.add_argument("--width", type=int, default=640)
p.add_argument("--height", type=int, default=480)
p.add_argument("--use-tpu", action="store_true", help="Fuerza USE_TPU=true en el entorno")
p.add_argument("--label", default="run", help="Nombre corto de la corrida (cpu_1cam, …)")
p.add_argument("--escena", default="", help="Descripción de la escena (ej. '1 persona sentada frente a cámara')")
p.add_argument("--out", default=None, help="Carpeta de resultados (por defecto <code-dir>/results)")
args = p.parse_args()

CODE_DIR = os.path.abspath(os.path.expanduser(args.code_dir))
OUT_DIR = os.path.abspath(os.path.expanduser(args.out or os.path.join(CODE_DIR, "results")))
CAMS = [int(c) if c.isdigit() else c for c in args.cams]
if not os.path.isfile(os.path.join(CODE_DIR, "app", "ai_engine.py")):
    sys.exit(f"[baseline] ERROR: {CODE_DIR} no contiene el código de PiVision (falta app/ai_engine.py). "
             f"¿Corriste tools/prepare_baseline.sh?")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[baseline] {msg}", file=sys.__stderr__, flush=True)


# ---------------------------------------------------------------------------
# Entorno: el código medido se importa desde CODE_DIR, con su .env
# ---------------------------------------------------------------------------
os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)
if args.use_tpu:
    os.environ["USE_TPU"] = "true"  # load_dotenv no sobrescribe variables ya definidas
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(CODE_DIR, ".env"))
except Exception:
    pass

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import cv2  # noqa: E402


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def pkg(name):
    try:
        return md.version(name)
    except Exception:
        return None


def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None


RUN_START = dt.datetime.now().isoformat(sep=" ")


def db_counts():
    import sqlite3
    path = os.path.join(CODE_DIR, "database", "pivision.db")
    out = {"db_path": path, "existe": os.path.exists(path)}
    if not out["existe"]:
        return out
    try:
        con = sqlite3.connect(path)
        has_events = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'").fetchone()
        if has_events:   # Fase 1 en adelante: tabla única de eventos
            out["tabla"] = "events"
            out["eventos_por_tipo"] = dict(con.execute(
                "SELECT type, COUNT(*) FROM events WHERE source != 'legacy' GROUP BY type").fetchall())
            out["eventos_en_prueba"] = dict(con.execute(
                "SELECT type, COUNT(*) FROM events WHERE source != 'legacy' AND ts >= ? GROUP BY type",
                (RUN_START,)).fetchall())
        else:            # código original
            out["tabla"] = "detection_events"
            out["eventos_por_tipo"] = dict(con.execute(
                "SELECT event_type, COUNT(*) FROM detection_events GROUP BY event_type").fetchall())
        out["rostros_conocidos"] = con.execute("SELECT COUNT(*) FROM known_faces").fetchone()[0]
        con.close()
    except Exception as e:
        out["error"] = str(e)
    return out


# ---------------------------------------------------------------------------
# Metadatos del entorno
# ---------------------------------------------------------------------------
meta = {
    "label": args.label,
    "escena": args.escena,
    "fecha": dt.datetime.now().isoformat(timespec="seconds"),
    "code_dir": CODE_DIR,
    "git_commit": sh(f"git -C '{CODE_DIR}' rev-parse --short HEAD"),
    "git_describe": sh(f"git -C '{CODE_DIR}' describe --tags --always --dirty"),
    "git_cambios_sin_commit": sh(f"git -C '{CODE_DIR}' status --porcelain --untracked-files=no").splitlines(),
    "camaras": [str(c) for c in CAMS],
    "resolucion_pipeline": f"{args.width}x{args.height}",
    "duracion_s": args.duration,
    "warmup_s": args.warmup,
    "env": {k: os.getenv(k) for k in ["USE_TPU", "DETECTION_FRAME_SKIP",
                                      "FACE_MATCH_THRESHOLD", "GESTURE_DETECTION_CONFIDENCE"]},
    "sistema": {
        "modelo_pi": sh("tr -d '\\0' < /proc/device-tree/model"),
        "os": sh("grep PRETTY_NAME /etc/os-release | cut -d= -f2 | tr -d '\"'"),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "nucleos": psutil.cpu_count(),
        "ram_total_mb": round(psutil.virtual_memory().total / 2**20),
        "hailo_dev": os.path.exists("/dev/hailo0"),
        "throttled_inicio": sh("vcgencmd get_throttled"),
    },
    "paquetes": {n: pkg(n) for n in ["opencv-contrib-python", "opencv-python", "mediapipe",
                                     "numpy", "tflite-runtime", "ai-edge-litert", "flask"]},
    "modelo_mobilefacenet_sha256": sha256(os.path.join(CODE_DIR, "models", "mobilefacenet.tflite")),
}

# ---------------------------------------------------------------------------
# Instrumentación
# ---------------------------------------------------------------------------
T0 = [None]                 # inicio de la ventana válida (después del warmup)
samples = defaultdict(list)  # etapa -> [(t, ms)]
counters = defaultdict(int)  # contadores varios
frames = defaultdict(list)   # cam_id -> [t de cada frame procesado]


def timed(stage, key_fn=None):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            t = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                end = time.perf_counter()
                samples[stage].append((end, (end - t) * 1000.0))
                if key_fn:
                    key_fn(a, end)
        return wrapper
    return deco


def patch(module_name, class_name, method, stage, key_fn=None):
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        setattr(cls, method, timed(stage, key_fn)(getattr(cls, method)))
        return True
    except Exception as e:
        log(f"no se instrumentó {module_name}.{class_name}.{method}: {e}")
        return False


def count_init(module_name, class_name, counter):
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        orig = cls.__init__

        @functools.wraps(orig)
        def init(self, *a, **kw):
            counters[counter] += 1
            return orig(self, *a, **kw)
        cls.__init__ = init
    except Exception as e:
        log(f"no se contó {module_name}.{class_name}: {e}")


def on_frame(a, end):
    frames[str(getattr(a[0], "cam_id", "?"))].append(end)


STAGES = [
    ("app.frame_processor", "FrameProcessor", "detect_motion", "movimiento"),
    ("app.face_mesh_processor", "FaceMeshProcessor", "process", "malla_facial"),
    ("app.face_normalizer", "FaceNormalizer", "normalize", "normalizacion"),
    ("face_encoder", "FaceEncoder", "encode", "embedding"),
    ("face_matcher", "FaceMatcher", "match", "comparacion"),
    ("app.gesture_handler", "GestureHandler", "analyze", "gestos"),
    ("person_tracker", "CentroidTracker", "update", "tracking"),
    ("app.object_detector", "ObjectDetector", "detect", "objetos_hailo"),   # Fase 3+ (no existe en el original)
]
# Inferencias reales de MediaPipe (excluye los frames que se sirven de caché)
STAGES += [
    ("mediapipe.python.solutions.face_mesh", "FaceMesh", "process", "mp_facemesh_inferencia"),
    ("mediapipe.python.solutions.holistic", "Holistic", "process", "mp_holistic_inferencia"),
]
instrumented = [s[3] for s in STAGES if patch(*s)]
patch("app.ai_engine", "AIPipeline", "process", "pipeline_total", key_fn=on_frame)
count_init("face_encoder", "FaceEncoder", "interpretes_mobilefacenet_creados")
count_init("face_matcher", "FaceMatcher", "facematcher_creados")

# ¿Se usa realmente el cargador con acelerador (USE_TPU)?
try:
    ai = importlib.import_module("app.ai_engine")
    if hasattr(ai, "load_tflite_model"):
        orig_loader = ai.load_tflite_model

        def loader(*a, **kw):
            counters["load_tflite_model_llamadas"] += 1
            return orig_loader(*a, **kw)
        ai.load_tflite_model = loader
        counters["load_tflite_model_llamadas"] += 0
    # Desactivar notificaciones reales durante la prueba
    if hasattr(ai, "send_notification_if_gesture"):
        def no_notify(*a, **kw):
            counters["notificaciones_suprimidas"] += 1
        ai.send_notification_if_gesture = no_notify
except Exception as e:
    log(f"no se pudo preparar app.ai_engine: {e}")

# ---------------------------------------------------------------------------
# 1) FPS nativo de cada cámara, sin IA
# ---------------------------------------------------------------------------
raw_fps = {}
for c in CAMS:
    cap = cv2.VideoCapture(c)
    if not cap.isOpened():
        raw_fps[str(c)] = {"error": "no abre"}
        continue
    n, t_end, shape = 0, time.time() + 5, None
    while time.time() < t_end:
        ok, fr = cap.read()
        if ok:
            n += 1
            shape = fr.shape
    cap.release()
    raw_fps[str(c)] = {"fps": round(n / 5, 1), "frame": f"{shape[1]}x{shape[0]}" if shape else None}
    log(f"cámara {c} sin IA: {raw_fps[str(c)]}")
time.sleep(0.5)

# ---------------------------------------------------------------------------
# 2) Pipeline real
# ---------------------------------------------------------------------------
db_before = db_counts()
from app.camera_manager import Camera  # noqa: E402

pipeline_log = io.StringIO()
cams = []
ram_before = psutil.Process().memory_info().rss
with contextlib.redirect_stdout(pipeline_log):
    for c in CAMS:
        cam = Camera(c, args.width, args.height)
        cams.append(cam)
ram_models = psutil.Process().memory_info().rss - ram_before

sys_samples = []
stop = threading.Event()


def sampler():
    proc = psutil.Process()
    psutil.cpu_percent(None)
    proc.cpu_percent(None)
    while not stop.wait(1.0):
        sys_samples.append({
            "t": time.perf_counter(),
            "cpu_total": psutil.cpu_percent(None),
            "cpu_por_nucleo": psutil.cpu_percent(None, percpu=True),
            "cpu_proceso": proc.cpu_percent(None),
            "rss_mb": proc.memory_info().rss / 2**20,
            "ram_sistema_pct": psutil.virtual_memory().percent,
            "temp_c": read_temp(),
            "freq_mhz": (psutil.cpu_freq().current if psutil.cpu_freq() else None),
        })


with contextlib.redirect_stdout(pipeline_log):
    for cam in cams:
        cam.start()
    threading.Thread(target=sampler, daemon=True).start()
    log(f"warmup {args.warmup}s …")
    time.sleep(args.warmup)
    T0[0] = time.perf_counter()
    log(f"midiendo {args.duration}s …")
    time.sleep(args.duration)
    T1 = time.perf_counter()
    stop.set()
    for cam in cams:
        cam.stop()
    time.sleep(0.5)

# Fase 1+: esperar a que la capa de eventos termine de escribir
_es = sys.modules.get("app.event_store")
if _es is not None and getattr(_es, "_store", None) is not None:
    _es._store.flush()
db_after = db_counts()

# ---------------------------------------------------------------------------
# 3) Resumen
# ---------------------------------------------------------------------------
def in_window(lst):
    return [ms for (t, ms) in lst if T0[0] <= t <= T1]


def stats(vals):
    if not vals:
        return {"n": 0}
    a = np.array(vals)
    return {"n": int(a.size), "media_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


win = T1 - T0[0]
total_frames = sum(len([t for t in v if T0[0] <= t <= T1]) for v in frames.values())
fps = {cam: round(len([t for t in v if T0[0] <= t <= T1]) / win, 2) for cam, v in frames.items()}

stage_stats = {}
pipe_total_ms = sum(in_window(samples["pipeline_total"])) or 1.0
for stage in instrumented + ["pipeline_total"]:
    vals = in_window(samples[stage])
    s = stats(vals)
    s["llamadas_por_frame"] = round(len(vals) / total_frames, 2) if total_frames else None
    s["pct_tiempo_pipeline"] = round(100 * sum(vals) / pipe_total_ms, 1) if stage != "pipeline_total" else 100.0
    stage_stats[stage] = s

ss = [s for s in sys_samples if T0[0] <= s["t"] <= T1]


def agg(key):
    v = [s[key] for s in ss if s[key] is not None]
    return {"media": round(float(np.mean(v)), 1), "max": round(float(np.max(v)), 1)} if v else None


eventos_nuevos = {}
if "eventos_en_prueba" in db_after:      # Fase 1+: cuenta por fecha, no por diferencia de tablas
    eventos_nuevos = db_after["eventos_en_prueba"]
for k in [] if eventos_nuevos else set(db_after.get("eventos_por_tipo", {})) | set(db_before.get("eventos_por_tipo", {})):
    eventos_nuevos[k] = db_after.get("eventos_por_tipo", {}).get(k, 0) - db_before.get("eventos_por_tipo", {}).get(k, 0)

log_text = pipeline_log.getvalue()
result = {
    "meta": meta,
    "fps_camara_sin_ia": raw_fps,
    "fps_pipeline_por_camara": fps,
    "fps_pipeline_total": round(total_frames / win, 2),
    "etapas": stage_stats,
    "sistema": {
        "cpu_total_pct": agg("cpu_total"),
        "cpu_proceso_pct": agg("cpu_proceso"),
        "rss_proceso_mb": agg("rss_mb"),
        "ram_sistema_pct": agg("ram_sistema_pct"),
        "temp_c": agg("temp_c"),
        "freq_mhz": agg("freq_mhz"),
        "nucleo_mas_cargado_pct": (round(float(np.max([np.mean([s["cpu_por_nucleo"][i] for s in ss])
                                                        for i in range(len(ss[0]["cpu_por_nucleo"]))])), 1)
                                   if ss else None),
        "ram_carga_modelos_mb": round(ram_models / 2**20, 1),
        "throttled_fin": sh("vcgencmd get_throttled"),
    },
    "contadores": dict(counters),
    "bd": {"antes": db_before, "despues": db_after, "eventos_guardados_en_prueba": eventos_nuevos},
    "log_pipeline": {
        "lineas": len(log_text.splitlines()),
        "menciona_acelerador": [l for l in log_text.splitlines() if "IA Hat" in l or "TPU" in l][:5],
    },
}

stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
base = os.path.join(OUT_DIR, f"baseline_{args.label}_{stamp}")
with open(base + ".json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
with open(base + "_pipeline.log", "w") as f:
    f.write(log_text)

# Markdown para el informe
L = []
L.append(f"# Baseline PiVision IA — {args.label}\n")
L.append(f"- Fecha: {meta['fecha']}  |  Código: `{meta['git_describe']}`  |  Escena: {args.escena or '—'}")
L.append(f"- Equipo: {meta['sistema']['modelo_pi']} · {meta['sistema']['os']} · Python {meta['sistema']['python']}")
L.append(f"- Cámaras: {', '.join(meta['camaras'])} · Resolución del pipeline {meta['resolucion_pipeline']} · "
         f"{args.duration}s medidos tras {args.warmup}s de warmup")
L.append(f"- Variables: " + ", ".join(f"{k}={v}" for k, v in meta["env"].items()) + "\n")
L.append("## FPS\n")
L.append("| Cámara | Sin IA | Con pipeline |\n|---|---|---|")
for c in meta["camaras"]:
    L.append(f"| {c} | {raw_fps.get(c, {}).get('fps', '—')} | {fps.get(c, 0)} |")
L.append(f"\nFPS total del sistema: **{result['fps_pipeline_total']}**\n")
L.append("## Latencia por etapa\n")
L.append("| Etapa | Llamadas/frame | Media ms | p50 | p95 | Máx | % del pipeline |\n|---|---|---|---|---|---|---|")
for st, s in stage_stats.items():
    if s.get("n"):
        L.append(f"| {st} | {s['llamadas_por_frame']} | {s['media_ms']} | {s['p50_ms']} | {s['p95_ms']} | "
                 f"{s['max_ms']} | {s['pct_tiempo_pipeline']} |")
    else:
        L.append(f"| {st} | 0 | — | — | — | — | — |")
S = result["sistema"]
L.append("\n## Recursos\n")
L.append("| Métrica | Media | Máx |\n|---|---|---|")
for k, name in [("cpu_total_pct", "CPU total %"), ("cpu_proceso_pct", "CPU del proceso % (400 = 4 núcleos)"),
                ("rss_proceso_mb", "RAM del proceso MB"), ("ram_sistema_pct", "RAM del sistema %"),
                ("temp_c", "Temperatura °C"), ("freq_mhz", "Frecuencia CPU MHz")]:
    v = S.get(k)
    L.append(f"| {name} | {v['media'] if v else '—'} | {v['max'] if v else '—'} |")
L.append(f"\n- Núcleo más cargado (media): {S['nucleo_mas_cargado_pct']} %")
L.append(f"- RAM usada al cargar modelos: {S['ram_carga_modelos_mb']} MB")
L.append(f"- Throttling: inicio `{meta['sistema']['throttled_inicio'] or '—'}` · fin `{S['throttled_fin'] or '—'}`\n")
L.append("## Evidencia de hallazgos\n")
C = result["contadores"]
L.append(f"- Intérpretes MobileFaceNet creados: **{C.get('interpretes_mobilefacenet_creados', 0)}** "
         f"para {len(CAMS)} cámara(s)")
L.append(f"- `load_tflite_model()` (ruta con acelerador / USE_TPU) llamada: **{C.get('load_tflite_model_llamadas', 0)}** veces")
emb = stage_stats.get("embedding", {})
mesh = stage_stats.get("mp_facemesh_inferencia", {})
L.append(f"- Inferencias reales de malla facial por frame: {mesh.get('llamadas_por_frame')} "
         f"(DETECTION_FRAME_SKIP={meta['env'].get('DETECTION_FRAME_SKIP')}) · embeddings por frame: "
         f"{emb.get('llamadas_por_frame')}")
L.append(f"- Eventos guardados en BD durante la prueba: {eventos_nuevos or 'ninguno'}")
L.append(f"- Rostros conocidos en BD: {db_after.get('rostros_conocidos', '—')}")
L.append(f"- Notificaciones suprimidas: {C.get('notificaciones_suprimidas', 0)}")
with open(base + ".md", "w") as f:
    f.write("\n".join(L) + "\n")

log(f"listo → {base}.md")
print("\n".join(L), file=sys.__stdout__)
os._exit(0)  # los hilos de MediaPipe a veces no terminan solos
