# app/ai_engine.py

import time
from collections import Counter, deque
import cv2
import os
from dotenv import load_dotenv
import tflite_runtime.interpreter as tflite

from app.frame_processor      import FrameProcessor
from app.face_mesh_processor  import FaceMeshProcessor
from app.face_normalizer      import FaceNormalizer
from app.gesture_handler      import GestureHandler
from app.event_store          import get_store
from app.object_detector      import ObjectDetector, nombre as nombre_objeto
from hailo_service.pose       import gestos as gestos_pose, SKELETON
from face_encoder             import FaceEncoder
from face_matcher             import FaceMatcher
from db                       import Database
from person_tracker           import CentroidTracker
from app.system_monitor       import get_system_metrics
from utils.notifier           import send_notification_if_gesture

# Cargar configuración desde .env
load_dotenv()

# Reglas anti-ruido para guardar eventos (Fase 1). Ajustables en .env
MOTION_MIN_AREA_PCT   = float(os.getenv("MOTION_MIN_AREA_PCT", 0.5))   # % del cuadro que debe cambiar
MOTION_EVENT_INTERVAL = float(os.getenv("MOTION_EVENT_INTERVAL", 5))   # s entre eventos de movimiento
FACE_EVENT_INTERVAL   = float(os.getenv("FACE_EVENT_INTERVAL", 10))    # s entre eventos de la misma persona
COUNT_EVENT_INTERVAL  = float(os.getenv("COUNT_EVENT_INTERVAL", 2))    # s mínimos entre cambios de conteo
COUNT_STABLE_S        = float(os.getenv("COUNT_STABLE_S", 1.0))        # ventana para suavizar el conteo
EVENT_SNAPSHOTS       = os.getenv("EVENT_SNAPSHOTS", "all").lower()    # all | gesture | none

# Detección de objetos con el Hailo (Fase 3)
OBJECT_DETECTION      = os.getenv("OBJECT_DETECTION", "true").lower() == "true"
OBJECT_CLASSES        = [c.strip() for c in os.getenv("OBJECT_CLASSES",
                         "person,car,motorcycle,backpack,dog,cell phone").split(",") if c.strip()]
OBJECT_MIN_SCORE      = float(os.getenv("OBJECT_MIN_SCORE", 0.5))
# Umbrales por clase "clase:valor,clase:valor". Objetos pequeños o parcialmente tapados
# (mochila puesta, celular en la mano) salen con menos confianza que una persona o un auto.
OBJECT_MIN_SCORES     = {k.strip(): float(v) for k, v in
                         (p.split(":") for p in os.getenv("OBJECT_MIN_SCORES",
                          "backpack:0.35,cell phone:0.55").split(",") if ":" in p)}
# Persistencia: un objeto cuenta solo si aparece en K de los últimos N cuadros analizados.
# Los falsos positivos (una mano, una oreja) parpadean; un objeto real se sostiene.
OBJECT_PERSIST_N      = max(1, int(os.getenv("OBJECT_PERSIST_N", 5)))
OBJECT_PERSIST_K      = max(1, min(OBJECT_PERSIST_N, int(os.getenv("OBJECT_PERSIST_K", 3))))
# Cada cuántos cuadros se corre el modelo de objetos. "auto": 1 sin pose; 5 con pose en el
# Hailo (la pose corre en cada cuadro y da personas y gestos; así el chip casi no cambia de modelo)
_ode = os.getenv("OBJECT_DETECT_EVERY", "auto").strip().lower()
OBJECT_DETECT_EVERY   = None if _ode == "auto" else max(1, int(_ode))
OBJECT_EVENT_INTERVAL = float(os.getenv("OBJECT_EVENT_INTERVAL", 10))     # s entre eventos de la misma clase

# Gestos con pose en el Hailo (Fase 5). Si el servicio no tiene pose, se usa Holistic (CPU).
POSE_GESTURES         = os.getenv("POSE_GESTURES", "true").lower() == "true"
POSE_MIN_SCORE        = float(os.getenv("POSE_MIN_SCORE", 0.5))   # confianza mínima de la persona
POSE_MIN_VIS          = float(os.getenv("POSE_MIN_VIS", 0.5))     # visibilidad mínima de hombros y muñecas
GESTURE_COOLDOWN      = float(os.getenv("GESTURE_COOLDOWN", 1.0)) # s entre gestos (igual que el original)

# Hallazgo 7: el embedding (MobileFaceNet, ~21 ms) solo se recalcula cuando la malla facial
# se recalculó y, mientras sigan las mismas caras, como máximo cada FACE_REENCODE_S segundos.
FACE_REENCODE_S       = float(os.getenv("FACE_REENCODE_S", 1.0))


def load_tflite_model(model_path):
    # Se conserva para la medición "antes" (tools/baseline.py). No se usa en el pipeline;
    # la aceleración real va por el servicio Hailo.
    use_tpu = os.getenv("USE_TPU", "false").lower() == "true"
    try:
        if use_tpu:
            from tflite_runtime.interpreter import load_delegate
            print("[INFO] Intentando usar IA Hat+ (Edge TPU)...")
            interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
            print("[INFO] Inferencia acelerada activada con IA Hat+")
        else:
            print("[INFO] Modo CPU activado para inferencia")
            interpreter = tflite.Interpreter(model_path=model_path)
    except Exception as e:
        print(f"[WARNING] Error al cargar IA Hat+ ({e}), usando CPU por defecto")
        interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    return interpreter


class EventPolicy:
    """Decide cuándo un evento repetido merece guardarse (evita 15 registros por segundo)."""
    def __init__(self):
        self._last = {}

    def allow(self, key, interval, now=None):
        now = now or time.time()
        if now - self._last.get(key, 0.0) >= interval:
            self._last[key] = now
            return True
        return False


class AIPipeline:
    def __init__(self, cam_id, width, height):
        self.cam_id  = cam_id
        self.width   = width
        self.height  = height

        self.motion_proc     = FrameProcessor()
        self.mesh_processor  = FaceMeshProcessor()
        self.normalizer      = FaceNormalizer()
        self.encoder         = FaceEncoder()

        # Umbral dinámico desde .env
        threshold = float(os.getenv("FACE_MATCH_THRESHOLD", 0.6))
        self.matcher         = FaceMatcher(Database(), threshold=threshold)

        # 80 px: con el Hailo se siguen cuerpos, que se desplazan más que una cara entre cuadros
        self.tracker         = CentroidTracker(max_disappeared=30, max_distance=80)
        # Holistic (CPU) solo se carga si hace falta: con pose en el Hailo no se usa
        self._gesture_handler = None

        self.store      = get_store()
        self.policy     = EventPolicy()
        self.last_count = None
        self._last_mesh   = None     # último resultado de la malla (mismo objeto = vino de caché)
        self._face_cache  = []       # [(etiqueta, resultado del match)] por cara
        self._last_encode = 0.0
        self._count_hist = deque()   # (tiempo, conteo) del último COUNT_STABLE_S
        self._read_id   = 0

        self.track_hist = {}
        self.count_up   = set()
        self.count_down = set()

        # Objetos (Fase 3): cliente del servicio Hailo; si no está corriendo, se sigue sin objetos
        self.objects   = (ObjectDetector(classes=OBJECT_CLASSES, min_score=OBJECT_MIN_SCORE,
                                         min_scores=OBJECT_MIN_SCORES, name=f"cam{cam_id}")
                          if OBJECT_DETECTION else None)
        self._frame_n  = 0
        self.last_objs = []          # objetos (sin personas) del último frame analizado
        self.last_persons = None     # cuerpos de persona del Hailo; None = servicio no disponible
        self.last_poses   = None     # poses del Hailo; None = sin pose (se usa Holistic)
        self._pose_disponible = None # None = aún no se sabe si el servicio tiene modelo de pose
        self._obj_hist = deque(maxlen=OBJECT_PERSIST_N)   # clases vistas en los últimos N cuadros

    @property
    def gesture_handler(self):
        if self._gesture_handler is None:
            # El pipeline registra los gestos en el EventStore; el handler ya no escribe en la BD
            self._gesture_handler = GestureHandler("sessions", log_to_db=False)
        return self._gesture_handler

    def _gesto_pose(self, frame):
        """Gesto desde las poses del Hailo. Devuelve (gesto, ruta_foto) o (None, None)."""
        g = gestos_pose(self.last_poses, POSE_MIN_VIS)
        if not g or not self.policy.allow("gesture", GESTURE_COOLDOWN):
            return None, None
        os.makedirs("sessions", exist_ok=True)
        cam = "".join(ch if ch.isalnum() else "-" for ch in str(self.cam_id))[-20:]
        ruta = os.path.join("sessions", f"cam{cam}_{g[0]}_{int(time.time())}.jpg")
        cv2.imwrite(ruta, frame)   # síncrono: la notificación la lee de inmediato
        return g[0], ruta

    def _emit(self, events, type, frame=None, **kw):
        # Foto del evento (Fase 2): se guarda en el hilo escritor del EventStore
        if frame is not None and EVENT_SNAPSHOTS == "all":
            kw["frame"] = frame
        evt = self.store.record(self.cam_id, type, **kw)
        events.append(evt)

    def _detect_objects(self, frame, events):
        """
        Corre sobre el cuadro limpio (antes de dibujar). Con pose en el Hailo: pose en cada
        cuadro (personas, conteo, seguimiento, gestos) y objetos 1 de cada N cuadros.
        """
        self._frame_n += 1
        if self.objects is None:
            return
        con_pose = POSE_GESTURES and self._pose_disponible is not False
        cada = OBJECT_DETECT_EVERY or (5 if con_pose else 1)
        turno_obj = (self._frame_n % cada == 0)
        if not (turno_obj or con_pose):
            return
        dets = self.objects.detect(frame, pose=con_pose, pose_score=POSE_MIN_SCORE, objects=turno_obj)
        self.last_poses = self.objects.last_poses
        if dets is None:                       # servicio caído: conteo por rostros, gestos con Holistic
            self.last_persons, self.last_objs, self.last_poses = None, [], None
            return
        if con_pose:
            self._pose_disponible = self.last_poses is not None   # False = el servicio no tiene pose
        # Personas: de la pose si la hay (cada cuadro); si no, del modelo de objetos
        if self.last_poses is not None:
            self.last_persons = [{"class": "person", "score": p["score"], "bbox": p["bbox"]}
                                 for p in self.last_poses]
        elif turno_obj:
            self.last_persons = [d for d in dets if d["class"] == "person"]
        if not turno_obj:
            return
        # Una persona no es un "objeto": sus cuerpos alimentan el conteo y el seguimiento
        objs = [d for d in dets if d["class"] != "person"]
        self._obj_hist.append({d["class"] for d in objs})
        vistas = Counter(c for clases in self._obj_hist for c in clases)
        estables = {c for c, n in vistas.items() if n >= OBJECT_PERSIST_K}
        self.last_objs = [d for d in objs if d["class"] in estables]
        by_class = {}
        for d in self.last_objs:
            by_class.setdefault(d["class"], []).append(d)
        for cls, items in by_class.items():
            if not self.policy.allow(("object", cls), OBJECT_EVENT_INTERVAL):
                continue
            best = max(items, key=lambda d: d["score"])
            self._emit(events, "object", frame=frame, label=nombre_objeto(cls), value=best["score"],
                       data={"class": cls, "count": len(items), "bbox": best["bbox"],
                             "boxes": [d["bbox"] for d in items],
                             "hailo_ms": self.objects.last_timing.get("infer_ms")})

    def _draw_objects(self, frame):
        for p in self.last_poses or []:
            kp = p["keypoints"]
            for a, b in SKELETON:
                if kp[a][2] >= POSE_MIN_VIS and kp[b][2] >= POSE_MIN_VIS:
                    cv2.line(frame, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])),
                             (255, 200, 0), 2)
        for d in self.last_persons or []:
            x, y, bw, bh = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (80, 200, 80), 2)
            cv2.putText(frame, f"persona {d['score']:.2f}", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 80), 1)
        for d in self.last_objs:
            x, y, bw, bh = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (60, 60, 255), 2)
            cv2.putText(frame, f"{nombre_objeto(d['class'])} {d['score']:.2f}", (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 1)

    def process(self, frame):
        events = []
        h, w = frame.shape[:2]

        # Objetos con el Hailo (Fase 3), sobre el cuadro sin anotaciones
        self._detect_objects(frame, events)

        # Movimiento: solo cambios que cubren al menos MOTION_MIN_AREA_PCT del cuadro
        _, contours = self.motion_proc.detect_motion(frame)
        if contours:
            area_pct = 100.0 * sum(cv2.contourArea(c) for c in contours) / float(h * w)
            if area_pct >= MOTION_MIN_AREA_PCT and self.policy.allow("motion", MOTION_EVENT_INTERVAL):
                self._emit(events, "motion", frame=frame, value=round(area_pct, 2),
                           data={"n": len(contours), "area_pct": round(area_pct, 2)})

        # Rostros
        mesh_faces = self.mesh_processor.process(frame)
        face_count = 0
        rects = []
        # La malla devuelve el mismo objeto cuando se salta el cuadro (DETECTION_FRAME_SKIP)
        fresca = mesh_faces is not self._last_mesh
        self._last_mesh = mesh_faces
        now_f = time.time()
        recodificar = fresca and (len(mesh_faces) != len(self._face_cache)
                                  or now_f - self._last_encode >= FACE_REENCODE_S)
        nueva_cache = []

        for i, lm in enumerate(mesh_faces):
            xs, ys = [int(p[0]) for p in lm], [int(p[1]) for p in lm]
            # Recorta al cuadro: con la cara en el borde, la malla da puntos fuera de la imagen
            x0, y0 = max(0, min(xs)), max(0, min(ys))
            x1, y1 = min(w - 1, max(xs)), min(h - 1, max(ys))
            rect = (x0, y0, x1 - x0, y1 - y0)
            rects.append(rect)

            if recodificar:
                face = self.normalizer.normalize(frame, lm)
                if face is None:
                    label, mres = None, None
                else:
                    emb = self.encoder.encode(face)
                    mres = self.matcher.match(emb)
                    label = mres["name"] if mres else "Desconocido"
                nueva_cache.append((label, mres))
            else:
                label, mres = self._face_cache[i] if i < len(self._face_cache) else (None, None)
            if label is not None:
                face_count += 1
                if self.policy.allow(("face", label), FACE_EVENT_INTERVAL):
                    self._emit(events, "face", frame=frame, label=label,
                               value=(mres["distance"] if mres else None),
                               data={"match": bool(mres), "person_id": (mres["id"] if mres else None),
                                     "bbox": list(rect)})

        if recodificar:
            self._face_cache, self._last_encode = nueva_cache, now_f
        elif fresca and not mesh_faces:
            self._face_cache = []

        # Conteo de personas: por cuerpos del Hailo si está disponible (no depende de ver la cara);
        # si no, por rostros. Se suaviza con el valor más frecuente del último COUNT_STABLE_S:
        # una detección perdida en uno o dos cuadros no cambia el conteo.
        if self.last_persons is not None:
            count, fuente, cajas = len(self.last_persons), "hailo", [d["bbox"] for d in self.last_persons]
        else:
            count, fuente, cajas = face_count, "rostros", [list(r) for r in rects]
        now = time.time()
        self._count_hist.append((now, count))
        while self._count_hist and now - self._count_hist[0][0] > COUNT_STABLE_S:
            self._count_hist.popleft()
        freq = Counter(c for _, c in self._count_hist).most_common()
        top = max(n for _, n in freq)
        count = max(c for c, n in freq if n == top)          # empate: el mayor
        cajas = cajas if count == len(cajas) else cajas[:count]
        lleno = now - self._count_hist[0][0] >= COUNT_STABLE_S * 0.8
        if (lleno and count != self.last_count
                and self.policy.allow("people_count", COUNT_EVENT_INTERVAL)):
            self.last_count = count
            self._emit(events, "people_count", frame=frame, value=count,
                       data={"count": count, "fuente": fuente, "boxes": cajas})

        # Gestos: pose en el Hailo si está disponible; si no, Holistic en CPU
        if self.last_poses is not None:
            g_evt, image_path = self._gesto_pose(frame)
            fuente_gesto = "hailo_pose"
        else:
            result = self.gesture_handler.analyze(self.cam_id, frame)
            if isinstance(result, tuple):
                g_evt, image_path = result
            else:
                g_evt, image_path = result, None
            fuente_gesto = "holistic"

        if g_evt:
            datos_gesto = {"gesture": g_evt, "fuente": fuente_gesto}
            if fuente_gesto == "hailo_pose":
                quien = [p for p in self.last_poses if g_evt in gestos_pose([p], POSE_MIN_VIS)]
                if quien:
                    datos_gesto["bbox"] = quien[0]["bbox"]
                    datos_gesto["frame_size"] = [w, h]
            self._emit(events, "gesture", label=g_evt, image_path=image_path, data=datos_gesto)
            if image_path:
                send_notification_if_gesture(image_path)

        # Tracking y cruce de línea
        LINE = h // 2
        cv2.line(frame, (0, LINE), (w, LINE), (255, 0, 0), 2)

        # Seguimiento y cruces: por cuerpos (Hailo) cuando hay servicio, si no por rostros
        track_rects = [tuple(d["bbox"]) for d in self.last_persons] if self.last_persons is not None else rects
        objs = self.tracker.update(track_rects)
        for oid, (cx, cy) in objs.items():
            prev = self.track_hist.get(oid, cy)
            self.track_hist[oid] = cy
            if prev < LINE <= cy and oid not in self.count_down:
                self.count_down.add(oid)
                self._emit(events, "line_cross", frame=frame, label="down", data={"dir": "down", "track_id": int(oid)})
            elif prev > LINE >= cy and oid not in self.count_up:
                self.count_up.add(oid)
                self._emit(events, "line_cross", frame=frame, label="up", data={"dir": "up", "track_id": int(oid)})

            cv2.putText(frame, str(oid), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.rectangle(frame, (cx-5, cy-5), (cx+5, cy+5), (0, 255, 0), 1)

        for (x, y, w0, h0) in rects:
            cv2.rectangle(frame, (x, y), (x+w0, y+h0), (0, 255, 255), 1)

        self._draw_objects(frame)

        return frame, events

    def get_last_events(self):
        """Compatibilidad: eventos nuevos de esta cámara desde la última lectura. Ya no vacía nada global."""
        events = self.store.recent(since_id=self._read_id, camera_id=self.cam_id)
        if events:
            self._read_id = events[-1]["id"]
        return list(events) + [get_system_metrics()]
