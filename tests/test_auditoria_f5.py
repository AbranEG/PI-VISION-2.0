#!/usr/bin/env python3
"""
Pruebas de las correcciones de la auditoría en la Fase 5 (hallazgos 10 y 11).
Uso (venv de la app):  python tests/test_auditoria_f5.py
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OK, FAIL = 0, 0


def check(name, cond, detail=""):
    global OK, FAIL
    OK, FAIL = (OK + 1, FAIL) if cond else (OK, FAIL + 1)
    print(f"  [{'OK' if cond else 'FALLA'}] {name} {'' if cond else detail}")


import face_encoder  # noqa: E402
creados = [0]
_orig = face_encoder.FaceEncoder.__init__


def _contar(self, *a, **kw):
    creados[0] += 1
    return _orig(self, *a, **kw)


face_encoder.FaceEncoder.__init__ = _contar

print("Hallazgo 10: FrameProcessor sin intérprete inútil")
import app.frame_processor as fp  # noqa: E402
fp.FaceEncoder = face_encoder.FaceEncoder
p = fp.FrameProcessor()
check("crear FrameProcessor no carga MobileFaceNet", creados[0] == 0, creados[0])
import numpy as np  # noqa: E402
p.detect_motion(np.zeros((480, 640, 3), np.uint8))
p.detect_motion(np.full((480, 640, 3), 255, np.uint8))
check("detect_motion funciona sin el encoder", creados[0] == 0)

print("Hallazgo 11: rostros nuevos sin reiniciar")
import face_matcher as fm  # noqa: E402
from db import Database  # noqa: E402
d = tempfile.mkdtemp(prefix="pv_aud_")
dbp = os.path.join(d, "t.db")
con = sqlite3.connect(dbp)
con.executescript("""CREATE TABLE known_faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
    face_encoding TEXT NOT NULL, photo_path TEXT, is_authorized INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);""")
con.commit(); con.close()
m = fm.FaceMatcher(Database(dbp), threshold=0.5)
v = np.ones(192) / np.sqrt(192)
check("sin rostros registrados → None", m.match(v) is None)
con = sqlite3.connect(dbp)
con.execute("INSERT INTO known_faces (name, face_encoding) VALUES (?, ?)", ("Luis", json.dumps(v.tolist())))
con.commit(); con.close()
fm_next = m._next_check
check("no recarga antes de tiempo (consulta ligera cada 10 s)", m.match(v) is None)
m._next_check = 0          # simula que pasaron 10 s
r = m.match(v)
check("reconoce al recién registrado sin reiniciar", r is not None and r["name"] == "Luis", r)
check("sin cambios no vuelve a recargar", m.maybe_reload(force=False) is False)
shutil.rmtree(d)

print(f"\nResultado: {OK} OK, {FAIL} fallas")
sys.exit(1 if FAIL else 0)
