# app/object_detector.py
"""
Cliente del servicio Hailo (Fase 3): detección de objetos desde la app (Python 3.11).

- Mantiene una conexión por cámara al socket Unix del servicio.
- Si el servicio no está corriendo, devuelve None y la app sigue sin objetos;
  reintenta conectar cada RETRY_S segundos y avisa en consola solo cuando cambia el estado.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hailo_service.protocol import DEFAULT_SOCKET, connect, recv_msg, send_msg  # noqa: E402

# Nombres en español para mostrar y guardar como etiqueta del evento
NOMBRES = {
    "person": "persona", "car": "auto", "motorcycle": "motocicleta", "backpack": "mochila", "dog": "perro",
    "bicycle": "bicicleta", "truck": "camión", "bus": "autobús", "cat": "gato", "handbag": "bolso",
    "suitcase": "maleta", "cell phone": "celular", "laptop": "laptop", "knife": "cuchillo",
}

RETRY_S = 5.0


def nombre(clase):
    return NOMBRES.get(clase, clase)


class ObjectDetector:
    def __init__(self, socket_path=DEFAULT_SOCKET, classes=None, min_score=0.5, timeout=1.0, name="cam",
                 min_scores=None):
        self.socket_path = socket_path
        self.classes = list(classes) if classes else None
        self.min_score = float(min_score)
        # Umbral por clase (p. ej. {"backpack": 0.35}); las demás usan min_score
        self.min_scores = {k: float(v) for k, v in (min_scores or {}).items()}
        self._req_score = min([self.min_score] + list(self.min_scores.values()))
        self.timeout = timeout
        self.name = name
        self._sock = None
        self._next_try = 0.0
        self.available = None      # None = aún no se intenta
        self.last_timing = {}
        self.last_poses = None     # poses del último cuadro; None = no se pidieron o no hay modelo
        self.stats = {"ok": 0, "fallas": 0, "reconexiones": 0}

    def _ensure(self):
        if self._sock is not None:
            return True
        now = time.time()
        if now < self._next_try:
            return False
        self._next_try = now + RETRY_S
        try:
            self._sock = connect(self.socket_path, timeout=self.timeout)
            self.stats["reconexiones"] += 1
            if not self.available:
                print(f"[Objetos {self.name}] Conectado al servicio Hailo ({self.socket_path})")
            self.available = True
            return True
        except OSError:
            if self.available is not False:
                print(f"[Objetos {self.name}] Servicio Hailo no disponible en {self.socket_path}; "
                      f"se sigue sin detección de objetos (reintento cada {RETRY_S:.0f} s)")
            self.available = False
            self._sock = None
            return False

    def _drop(self):
        try:
            if self._sock:
                self._sock.close()
        finally:
            self._sock = None

    def detect(self, frame_bgr, pose=False, pose_score=0.5, objects=True):
        """
        Lista de {"class", "score", "bbox": [x, y, w, h]} o None si el servicio no responde.
        Con pose=True, el mismo viaje trae las poses en self.last_poses (Fase 5).
        """
        self.last_poses = None
        if not self._ensure():
            return None
        try:
            send_msg(self._sock, {"op": "detect", "shape": list(frame_bgr.shape),
                                  "min_score": self._req_score, "classes": self.classes,
                                  "pose": bool(pose), "pose_score": float(pose_score),
                                  "objects": bool(objects)},
                     frame_bgr.tobytes() if frame_bgr.flags["C_CONTIGUOUS"] else frame_bgr.copy().tobytes())
            resp, _ = recv_msg(self._sock)
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "error del servicio"))
            self.stats["ok"] += 1
            self.last_timing = resp.get("timing", {})
            # Último tiempo conocido de cada modelo, para mostrarlo en el panel de estado
            self._ultimo_tiempo = {**getattr(self, "_ultimo_tiempo", {}),
                                   **{k: v for k, v in self.last_timing.items()
                                      if k in ("infer_ms", "pose_infer_ms", "pose_post_ms")}}
            self.last_poses = resp.get("poses") if pose else None
            dets = resp.get("detections", [])
            return [d for d in dets if d["score"] >= self.min_scores.get(d["class"], self.min_score)]
        except Exception as e:
            self.stats["fallas"] += 1
            print(f"[Objetos {self.name}] Falla con el servicio Hailo: {e}")
            self._drop()
            self.available = False
            self._next_try = time.time() + RETRY_S
            return None

    def close(self):
        self._drop()
