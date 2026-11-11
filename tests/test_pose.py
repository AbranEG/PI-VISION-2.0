#!/usr/bin/env python3
"""
Pruebas del decodificador de pose y del gesto "manos arriba" (Fase 5).
Uso:  python tests/test_pose.py     (venv de la app o Python del sistema; no necesita el Hailo)
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import numpy as np  # noqa: E402
from hailo_service.pose import decode_pose, gestos  # noqa: E402
from test_hailo_service import fake_pose_outputs  # noqa: E402

OK, FAIL = 0, 0


def check(name, cond, detail=""):
    global OK, FAIL
    OK, FAIL = (OK + 1, FAIL) if cond else (OK, FAIL + 1)
    print(f"  [{'OK' if cond else 'FALLA'}] {name} {'' if cond else detail}")


TF = (1.0, 0, 80, 640, 480)   # cuadro 640x480 con 80 px de relleno arriba
print("Decodificación de yolov8s_pose")
p = decode_pose(fake_pose_outputs(), TF)
check("una persona", len(p) == 1, p)
check("caja en píxeles del cuadro", p[0]["bbox"] == [248, 136, 160, 304], p[0]["bbox"])
check("hombro izquierdo", p[0]["keypoints"][5][:2] == [300.0, 220.0], p[0]["keypoints"][5])
check("muñeca derecha", p[0]["keypoints"][10][:2] == [370.0, 155.0], p[0]["keypoints"][10])
check("visibilidad como probabilidad", 0.9 < p[0]["keypoints"][10][2] < 1.0)
outs = fake_pose_outputs()
for k in ("conv71", "conv58", "conv44"):
    outs[k] = 1 / (1 + np.exp(-outs[k]))      # mismas salidas, pero ya como probabilidades
check("acepta scores ya en [0, 1]", decode_pose(outs, TF)[0]["bbox"] == [248, 136, 160, 304])
outs = fake_pose_outputs(); outs["conv58"][:] = -9.0
check("sin personas → lista vacía", decode_pose(outs, TF) == [])

print("Gesto manos arriba")
check("brazos arriba → manos_arriba", gestos(decode_pose(fake_pose_outputs(True), TF)) == ["manos_arriba"])
check("brazos abajo → nada", gestos(decode_pose(fake_pose_outputs(False), TF)) == [])
q = decode_pose(fake_pose_outputs(True), TF)
q[0]["keypoints"][9][2] = 0.2            # muñeca poco visible (antes Holistic la "inventaba")
check("muñeca poco visible → nada (hallazgo 14)", gestos(q) == [])
check("sin poses → nada", gestos(None) == [] and gestos([]) == [])

print(f"\nResultado: {OK} OK, {FAIL} fallas")
sys.exit(1 if FAIL else 0)
