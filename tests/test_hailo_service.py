#!/usr/bin/env python3
"""
Pruebas del servicio Hailo y su cliente (Fase 3).

    python tests/test_hailo_service.py            # Hailo simulado: protocolo, filtros, caídas y reconexión
    /usr/bin/python3 tests/test_hailo_service.py --real   # además, inferencia real en el Hailo-8 (en la Pi)

La versión simulada corre con el venv de la app (3.11); --real necesita el Python del sistema (3.13).
"""
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

OK, FAIL = 0, 0


def check(name, cond, detail=""):
    global OK, FAIL
    OK, FAIL = (OK + 1, FAIL) if cond else (OK, FAIL + 1)
    print(f"  [{'OK' if cond else 'FALLA'}]{' ' if cond else ''} {name} {'' if cond else detail}")


def fake_pose_outputs(arms_up=True):
    """Tensores crudos de yolov8s_pose con una persona: centro (328, 360) en 640x640, hombros en y=300."""
    outs = {}
    for H, (nb, ns, nk) in {20: ("conv70", "conv71", "conv72"), 40: ("conv57", "conv58", "conv59"),
                            80: ("conv43", "conv44", "conv45")}.items():
        outs[nb] = np.zeros((1, H, H, 64), np.float32)
        outs[nk] = np.zeros((1, H, H, 51), np.float32)
        outs[ns] = np.full((1, H, H, 1), -9.0, np.float32)
    stride, gx, gy = 16, 20, 22
    outs["conv58"][0, gy, gx, 0] = 3.0
    b = np.full((4, 16), -20.0, np.float32)
    for side, bin_ in enumerate([5, 9, 5, 10]):
        b[side, bin_] = 20.0
    outs["conv57"][0, gy, gx] = b.reshape(-1)
    cx, cy = (gx + .5) * stride, (gy + .5) * stride
    k = np.zeros((17, 3), np.float32)

    def put(i, x, y):
        k[i] = [((x - cx) / stride + .5) / 2, ((y - cy) / stride + .5) / 2, 3.0]
    for i in range(17):
        put(i, cx, cy)
    put(5, 300, 300); put(6, 360, 300)
    put(9, 290, 230 if arms_up else 380); put(10, 370, 235 if arms_up else 380)
    outs["conv59"][0, gy, gx] = k.reshape(-1)
    return outs


class FakeRunner:
    """Imita HailoRunner: objetos (persona 0.92 y perro 0.40) y, si se cargó, pose con manos arriba."""
    def __init__(self, hefs):
        self.models = {name: {"outs": ["nms"] if name == "detect" else [], "path": p} for name, p in hefs.items()}
        self.calls = 0

    def infer(self, name, rgb640):
        self.calls += 1
        if name == "pose":
            return fake_pose_outputs()
        out = [np.zeros((0, 5), np.float32) for _ in range(80)]
        out[0] = np.array([[0.25, 0.25, 0.75, 0.5, 0.92]], np.float32)   # persona
        out[16] = np.array([[0.6, 0.6, 0.7, 0.7, 0.40]], np.float32)     # perro, confianza baja
        return {"nms": [out]}

    def close(self):
        pass


def start_service(sock_path, fake=True, pose=True):
    import hailo_service.server as srv
    if fake:
        srv.HailoRunner = FakeRunner
        pose_hef = __file__ if pose else None      # cualquier archivo existente sirve de "HEF" simulado
    else:
        pose_hef = "/usr/share/hailo-models/yolov8s_pose_h8.hef" if pose else None
    svc = srv.Service(os.getenv("HAILO_HEF", "/usr/share/hailo-models/yolov8s_h8.hef"), sock_path,
                      pose_hef=pose_hef)
    th = threading.Thread(target=svc.serve, daemon=True)
    th.start()
    for _ in range(50):
        if os.path.exists(sock_path):
            break
        time.sleep(0.1)
    return svc, th


def load_client():
    """Carga solo app/object_detector.py (sin el paquete app, que arrastra Flask y MediaPipe)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("object_detector", os.path.join(ROOT, "app", "object_detector.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(real=False):
    od = load_client()
    ObjectDetector = od.ObjectDetector
    from hailo_service.protocol import connect, recv_msg, send_msg

    d = tempfile.mkdtemp(prefix="pv_hailo_")
    sock = os.path.join(d, "hailo.sock")
    frame = np.zeros((480, 640, 3), np.uint8)

    print("1. Servicio y protocolo" + (" (Hailo REAL)" if real else " (Hailo simulado)"))
    svc, th = start_service(sock, fake=not real)
    check("socket creado con permisos 600", oct(os.stat(sock).st_mode)[-3:] == "600")
    s = connect(sock)
    send_msg(s, {"op": "ping"}); r, _ = recv_msg(s); check("ping", r.get("pong") is True)
    send_msg(s, {"op": "info"}); r, _ = recv_msg(s); check("info con 80 clases", len(r.get("classes", [])) == 80)
    send_msg(s, {"op": "nada"}); r, _ = recv_msg(s); check("operación desconocida → error controlado", r["ok"] is False)
    s.close()

    print("2. Cliente de la app")
    det = ObjectDetector(socket_path=sock, classes=["person", "dog"], min_score=0.5, name="prueba")
    res = det.detect(frame)
    check("detecta", res is not None, res)
    if not real:
        check("filtra por confianza (perro 0.40 < 0.5 queda fuera)", [x["class"] for x in res] == ["person"], res)
        b = res[0]["bbox"]
        # 640x480 en lienzo 640x640: relleno de 80 px arriba; ymin 0.25 → 160-80 = 80 px
        check("recuadro en píxeles del cuadro original", b == [160, 80, 160, 320], b)
    check("tiempos reportados", "infer_ms" in det.last_timing, det.last_timing)
    if not real:
        det3 = ObjectDetector(socket_path=sock, classes=["person", "dog"], min_score=0.5,
                              min_scores={"dog": 0.35}, name="prueba3")
        r3 = det3.detect(frame)
        check("umbral por clase (perro 0.40 ≥ 0.35 entra; persona sigue con 0.5)",
              sorted(x["class"] for x in r3) == ["dog", "person"], r3)
    det2 = ObjectDetector(socket_path=sock, classes=["car"], min_score=0.1, name="prueba2")
    r2 = det2.detect(frame)
    check("filtro por clase", r2 is not None and all(x["class"] == "car" for x in r2), r2)

    n = 30
    t0 = time.perf_counter()
    for _ in range(n):
        det.detect(frame)
    ms = (time.perf_counter() - t0) * 1000 / n
    check(f"ida y vuelta por el socket: {ms:.1f} ms por frame", ms < (40 if real else 20))

    print("2b. Pose en el mismo viaje (Fase 5)")
    s2 = connect(sock)
    send_msg(s2, {"op": "info"}); r, _ = recv_msg(s2); s2.close()
    check("el servicio anuncia el modelo de pose", r.get("pose") is True, r.get("pose"))
    res = det.detect(frame, pose=True)
    check("devuelve poses junto con los objetos", res is not None and det.last_poses is not None, det.last_poses)
    if det.last_poses:
        p0 = det.last_poses[0]
        check("17 puntos por persona", len(p0["keypoints"]) == 17)
        check("tiempos de pose reportados", "pose_infer_ms" in det.last_timing, det.last_timing)
        if not real:
            check("caja de la persona en píxeles del cuadro", p0["bbox"] == [248, 136, 160, 304], p0["bbox"])
            import hailo_service.pose as hp
            check("gesto manos arriba desde la pose", hp.gestos(det.last_poses) == ["manos_arriba"])
    det.detect(frame, pose=False)
    check("sin pedir pose no se calcula", det.last_poses is None)

    print("3. Dos cámaras a la vez")
    errs = []

    def worker(k):
        dd = ObjectDetector(socket_path=sock, classes=["person"], name=f"hilo{k}")
        for _ in range(20):
            if dd.detect(frame) is None:
                errs.append(k)
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(2)]
    [t.start() for t in ths]; [t.join() for t in ths]
    check("40 peticiones concurrentes sin errores", not errs, errs)

    print("4. Caída y regreso del servicio")
    svc.stop(); th.join(timeout=5)
    check("con el servicio caído devuelve None (la app sigue)", det.detect(frame) is None)
    od.RETRY_S = 0.2
    det._next_try = 0
    svc, th = start_service(sock, fake=not real)
    time.sleep(0.3)
    check("se reconecta solo al volver el servicio", det.detect(frame) is not None)
    svc.stop(); th.join(timeout=5)

    print("5. Servicio sin modelo de pose")
    svc, th = start_service(sock, fake=not real, pose=False)
    det5 = ObjectDetector(socket_path=sock, classes=["person"], name="sinpose")
    det5.detect(frame, pose=True)
    check("pide pose pero el servicio no tiene: poses = None (la app usa Holistic)", det5.last_poses is None)
    svc.stop(); th.join(timeout=5)

    print(f"\nResultado: {OK} OK, {FAIL} fallas")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main(real="--real" in sys.argv))
