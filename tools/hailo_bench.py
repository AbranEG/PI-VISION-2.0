#!/usr/bin/python3
"""
hailo_bench.py — Fase 5: cuánto cuesta cada combinación de modelos en el Hailo-8.

Requiere el servicio corriendo:   /usr/bin/python3 -m hailo_service.server &
Uso:                              /usr/bin/python3 tools/hailo_bench.py [--frames 150] [--source 0|imagen.jpg]

Mide la ida y vuelta completa (app → servicio → Hailo → app) para:
  A) solo objetos en cada cuadro
  B) solo pose en cada cuadro
  C) objetos + pose en cada cuadro          (lo que hacía la primera versión de la Fase 5)
  D) pose en cada cuadro + objetos 1 de 5    (estrategia actual)
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hailo_service.protocol import DEFAULT_SOCKET, connect, recv_msg, send_msg  # noqa: E402


def stats(v):
    a = np.array(v)
    return round(float(a.mean()), 2), round(float(np.percentile(a, 95)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--source", default="0")
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    a = ap.parse_args()

    if os.path.isfile(a.source):
        frames = [cv2.imread(a.source)] * 30
    else:
        cap = cv2.VideoCapture(int(a.source), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frames = []
        for _ in range(60):
            ok, f = cap.read()
            if ok:
                frames.append(cv2.resize(f, (640, 480)))
        cap.release()
    if not frames:
        sys.exit("No se pudo leer la cámara (¿la app sigue corriendo?)")
    print(f"[bench] {len(frames)} cuadros de muestra · {a.frames} peticiones por modo")

    s = connect(a.socket, timeout=5)
    send_msg(s, {"op": "info"}); info, _ = recv_msg(s)
    if not info.get("pose"):
        print("[bench] Aviso: el servicio no tiene modelo de pose; B, C y D no son comparables")

    modos = {
        "A · solo objetos (cada cuadro)": lambda i: (True, False),
        "B · solo pose (cada cuadro)": lambda i: (False, True),
        "C · objetos + pose (cada cuadro)": lambda i: (True, True),
        "D · pose cada cuadro + objetos 1 de 5": lambda i: (i % 5 == 0, True),
    }
    filas = []
    for nombre, plan in modos.items():
        for i in range(10):     # calentamiento
            obj, pose = plan(i)
            send_msg(s, {"op": "detect", "shape": [480, 640, 3], "objects": obj, "pose": pose, "min_score": 0.5},
                     frames[i % len(frames)].tobytes()); recv_msg(s)
        tot, inf, pinf, ppost = [], [], [], []
        for i in range(a.frames):
            obj, pose = plan(i)
            f = frames[i % len(frames)]
            t0 = time.perf_counter()
            send_msg(s, {"op": "detect", "shape": list(f.shape), "objects": obj, "pose": pose, "min_score": 0.5},
                     f.tobytes())
            r, _ = recv_msg(s)
            tot.append((time.perf_counter() - t0) * 1000)
            t = r.get("timing", {})
            if obj:
                inf.append(t.get("infer_ms", 0))
            if pose:
                pinf.append(t.get("pose_infer_ms", 0)); ppost.append(t.get("pose_post_ms", 0))
        m, p95 = stats(tot)
        filas.append((nombre, m, p95, stats(inf)[0] if inf else None, stats(pinf)[0] if pinf else None,
                      stats(ppost)[0] if ppost else None))
    s.close()

    print("\n| Modo | Ida y vuelta media (ms) | p95 | Objetos en el Hailo | Pose en el Hailo | Decodificar pose |")
    print("|---|---|---|---|---|---|")
    for n, m, p95, i, pi, pp in filas:
        fmt = lambda x: "—" if x is None else x
        print(f"| {n} | **{m}** | {p95} | {fmt(i)} | {fmt(pi)} | {fmt(pp)} |")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                       f"hailo_bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump([dict(zip(["modo", "media_ms", "p95_ms", "objetos_ms", "pose_ms", "decod_pose_ms"], f))
                   for f in filas], fh, indent=2, ensure_ascii=False)
    print(f"\nResultados: {out}")


if __name__ == "__main__":
    main()
