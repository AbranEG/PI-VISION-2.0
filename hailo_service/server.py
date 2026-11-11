#!/usr/bin/python3
"""
Servicio de inferencia Hailo-8 para PiVision IA (Fase 3).

Corre con el Python 3.13 DEL SISTEMA (donde está hailo_platform):
    cd ~/PiVision-IA
    /usr/bin/python3 -m hailo_service.server
    /usr/bin/python3 -m hailo_service.server --hef /usr/share/hailo-models/yolov8s_h8.hef --socket /tmp/pivision-hailo.sock

La app (Python 3.11) se conecta por el socket Unix y envía frames BGR.
El servicio hace el ajuste a 640x640, la inferencia en el Hailo y devuelve
las detecciones en píxeles del frame original.

Varias cámaras pueden conectarse a la vez: cada conexión tiene su hilo y el
acceso al Hailo se serializa con un candado.
"""
import argparse
import os
import signal
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hailo_service.protocol import DEFAULT_SOCKET, recv_msg, send_msg   # noqa: E402
from hailo_service.detector import COCO, HailoRunner, letterbox, parse_nms_by_class   # noqa: E402
from hailo_service.pose import decode_pose   # noqa: E402


class Service:
    def __init__(self, hef, socket_path, pose_hef=None):
        self.hef = hef
        self.pose_hef = pose_hef if pose_hef and os.path.exists(pose_hef) else None
        self.socket_path = socket_path
        self.lock = threading.Lock()
        hefs = {"detect": hef}
        if self.pose_hef:
            hefs["pose"] = self.pose_hef
        try:
            self.runner = HailoRunner(hefs)
        except Exception as e:
            if len(hefs) == 1:
                raise
            # Si el Hailo no logra cargar los dos modelos, se sigue solo con objetos
            print(f"[hailo] No se pudo cargar el modelo de pose junto al de objetos ({e}); "
                  f"se continúa solo con objetos", flush=True)
            self.pose_hef = None
            self.runner = HailoRunner({"detect": hef})
        self.has_pose = "pose" in self.runner.models
        self.started = time.time()
        self.stats = {"peticiones": 0, "errores": 0, "infer_ms_total": 0.0, "pose_ms_total": 0.0,
                      "peticiones_pose": 0, "clientes": 0}
        self._stop = threading.Event()
        self._conns = set()

    # ------------------------------------------------------------------ inferencia
    def detect(self, frame_bgr, min_score=0.5, classes=None, pose=False, pose_score=0.5, objects=True):
        """
        Objetos y/o poses con un solo ajuste a 640x640. objects=False omite el modelo de
        objetos (la app lo corre solo 1 de cada N cuadros para que el chip no cambie de
        modelo en cada cuadro).
        """
        t0 = time.perf_counter()
        rgb, tf = letterbox(frame_bgr)
        t1 = time.perf_counter()
        dets = []
        if objects:
            with self.lock:
                outs = self.runner.infer("detect", rgb)
        t2 = time.perf_counter()
        if objects:
            raw = outs[self.runner.models["detect"]["outs"][0]]
            dets = parse_nms_by_class(raw, tf, min_score=min_score)
        wanted = set(classes) if classes else None
        out = []
        for cls_id, score, x1, y1, x2, y2 in dets:
            name = COCO[cls_id] if cls_id < len(COCO) else str(cls_id)
            if wanted and name not in wanted:
                continue
            out.append({"class": name, "score": round(score, 3), "bbox": [x1, y1, x2 - x1, y2 - y1]})
        t3 = time.perf_counter()
        timing = {"pre_ms": round((t1 - t0) * 1000, 2), "infer_ms": round((t2 - t1) * 1000, 2),
                  "post_ms": round((t3 - t2) * 1000, 2)}
        if objects:
            self.stats["infer_ms_total"] += timing["infer_ms"]
        poses = None
        if pose and self.has_pose:
            with self.lock:
                pouts = self.runner.infer("pose", rgb)
            t4 = time.perf_counter()
            poses = decode_pose(pouts, tf, score_thr=pose_score)
            t5 = time.perf_counter()
            timing["pose_infer_ms"] = round((t4 - t3) * 1000, 2)
            timing["pose_post_ms"] = round((t5 - t4) * 1000, 2)
            self.stats["pose_ms_total"] += timing["pose_infer_ms"]
            self.stats["peticiones_pose"] += 1
        return out, timing, poses

    # ------------------------------------------------------------------ red
    def handle(self, conn):
        self.stats["clientes"] += 1
        self._conns.add(conn)
        try:
            while not self._stop.is_set():
                try:
                    req, payload = recv_msg(conn)
                except (ConnectionError, OSError):
                    break
                op = req.get("op")
                try:
                    if op == "ping":
                        send_msg(conn, {"ok": True, "pong": True})
                    elif op == "info":
                        n = max(1, self.stats["peticiones"])
                        send_msg(conn, {"ok": True, "hef": self.hef, "pose_hef": self.pose_hef,
                                        "pose": self.has_pose, "classes": COCO,
                                        "uptime_s": round(time.time() - self.started),
                                        "peticiones": self.stats["peticiones"], "errores": self.stats["errores"],
                                        "infer_ms_promedio": round(self.stats["infer_ms_total"] / n, 2),
                                        "pose_ms_promedio": round(self.stats["pose_ms_total"]
                                                                  / max(1, self.stats["peticiones_pose"]), 2)})
                    elif op == "detect":
                        h, w, c = req["shape"]
                        frame = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c)
                        dets, timing, poses = self.detect(frame, float(req.get("min_score", 0.5)), req.get("classes"),
                                                          bool(req.get("pose")), float(req.get("pose_score", 0.5)),
                                                          bool(req.get("objects", True)))
                        self.stats["peticiones"] += 1
                        resp = {"ok": True, "detections": dets, "timing": timing}
                        if req.get("pose"):
                            resp["poses"] = poses     # None = el servicio no tiene modelo de pose
                        send_msg(conn, resp)
                    else:
                        send_msg(conn, {"ok": False, "error": f"operación desconocida: {op}"})
                except Exception as e:
                    self.stats["errores"] += 1
                    try:
                        send_msg(conn, {"ok": False, "error": str(e)})
                    except OSError:
                        break
        finally:
            self.stats["clientes"] -= 1
            self._conns.discard(conn)
            conn.close()

    def serve(self):
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)   # solo el usuario que corre la app
        srv.listen(8)
        srv.settimeout(1.0)
        print(f"[hailo] Servicio listo en {self.socket_path} · objetos: {os.path.basename(self.hef)} · "
              f"pose: {os.path.basename(self.pose_hef) if self.has_pose else 'no'}", flush=True)
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = srv.accept()
                except socket.timeout:
                    continue
                conn.settimeout(None)
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
        finally:
            srv.close()
            for c in list(self._conns):      # corta a los clientes: así detectan la caída
                try:
                    c.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)
            self.runner.close()
            print("[hailo] Servicio detenido", flush=True)

    def stop(self, *_):
        self._stop.set()


def main():
    ap = argparse.ArgumentParser(description="Servicio de inferencia Hailo-8 para PiVision IA")
    ap.add_argument("--hef", default=os.getenv("HAILO_HEF", "/usr/share/hailo-models/yolov8s_h8.hef"))
    ap.add_argument("--pose-hef", default=os.getenv("HAILO_POSE_HEF", "/usr/share/hailo-models/yolov8s_pose_h8.hef"))
    ap.add_argument("--sin-pose", action="store_true", help="no cargar el modelo de pose")
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    a = ap.parse_args()
    svc = Service(a.hef, a.socket, pose_hef=None if a.sin_pose else a.pose_hef)
    signal.signal(signal.SIGTERM, svc.stop)
    signal.signal(signal.SIGINT, svc.stop)
    svc.serve()


if __name__ == "__main__":
    main()
