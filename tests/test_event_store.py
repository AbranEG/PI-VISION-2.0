#!/usr/bin/env python3
"""
Pruebas de la capa de persistencia de eventos (Fase 1).
Uso:  python tests/test_event_store.py            (desde ~/PiVision-IA, con el venv activo)
No usa cámara ni modelos; trabaja sobre bases de datos temporales.
"""
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time

# Se carga solo app/event_store.py (sin importar el paquete app, que arrastra OpenCV y MediaPipe)
import importlib.util  # noqa: E402
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "event_store.py")
_spec = importlib.util.spec_from_file_location("event_store", _p)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EventStore, ensure_schema = _mod.EventStore, _mod.ensure_schema

OK, FAIL = 0, 0


def check(name, cond, detail=""):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FALLA] {name} {detail}")


def tmpdb():
    d = tempfile.mkdtemp(prefix="pv_test_")
    return d, os.path.join(d, "test.db")


def test_basico():
    print("1. Registro, persistencia y consulta")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.1)
    s.record(0, "motion", value=3.2, data={"n": 5, "area_pct": 3.2})
    s.record(0, "face", label="Desconocido", data={"match": False})
    s.record(0, "face", label="Luis", value=0.31, data={"match": True})
    s.record(1, "gesture", label="manos_arriba", image_path="sessions/x.jpg", data={"gesture": "manos_arriba"})
    s.record(0, "people_count", value=2, data={"count": 2})
    s.flush()
    check("5 eventos guardados", s.count() == 5, s.stats())
    check("filtro por tipo", s.count(type="face") == 2)
    check("filtro por varios tipos", s.count(type=["face", "gesture"]) == 3)
    check("filtro por cámara", s.count(camera_id=1) == 1)
    check("filtro por etiqueta", s.count(label="Luis") == 1)
    ev = s.query(type="face", label="Luis")[0]
    check("data JSON se recupera", ev["data"].get("match") is True and ev["value"] == 0.31)
    check("más reciente primero", s.query(limit=1)[0]["type"] == "people_count")
    try:
        s.record(0, "inventado")
        check("rechaza tipos desconocidos", False)
    except ValueError:
        check("rechaza tipos desconocidos", True)
    s.close()
    shutil.rmtree(d)


def test_vivo_sin_robo():
    print("2. Búfer en vivo: dos lectores no se roban eventos (hallazgo 5)")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.1)
    for i in range(10):
        s.record(0, "motion", data={"n": i})
    a = s.recent(since_id=0)
    b = s.recent(since_id=0)
    check("lector A ve 10", len(a) == 10)
    check("lector B también ve 10", len(b) == 10)
    s.record(0, "face", label="X")
    check("cursor: solo lo nuevo", [e["type"] for e in s.recent(since_id=a[-1]["id"])] == ["face"])
    check("filtro por cámara en vivo", len(s.recent(camera_id=1)) == 0)
    s.close()
    shutil.rmtree(d)


def test_concurrencia_y_velocidad():
    print("3. Concurrencia: 4 hilos escribiendo, 1 leyendo")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.2)
    N = 500

    def writer(cam):
        for i in range(N):
            s.record(cam, "motion", data={"i": i})

    t0 = time.perf_counter()
    ths = [threading.Thread(target=writer, args=(c,)) for c in range(4)]
    stop = threading.Event()
    reads = [0]

    def reader():
        while not stop.is_set():
            s.query(limit=50)
            reads[0] += 1

    r = threading.Thread(target=reader)
    r.start()
    [t.start() for t in ths]
    [t.join() for t in ths]
    t_rec = (time.perf_counter() - t0) * 1000 / (4 * N)
    s.flush(timeout=20)
    stop.set(); r.join()
    check(f"{4 * N} eventos guardados sin pérdidas", s.count() == 4 * N and s.stats()["perdidos"] == 0, s.stats())
    check(f"record() no bloquea: {t_rec:.3f} ms por evento", t_rec < 1.0)
    check(f"lecturas simultáneas sin error ({reads[0]} consultas)", reads[0] > 0)
    s.close()
    shutil.rmtree(d)


def test_resumen():
    print("4. Resumen por tipo, cámara y hora (base de la Fase 4)")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.1)
    s.record(0, "face", label="A", ts="2026-11-01 10:05:00.000000")
    s.record(0, "face", label="A", ts="2026-11-01 10:40:00.000000")
    s.record(1, "gesture", label="manos_arriba", ts="2026-11-01 11:10:00.000000")
    s.flush()
    r = s.summary(since="2026-11-01", until="2026-11-02")
    check("total 3", r["total"] == 3)
    check("por tipo", r["por_tipo"] == {"face": 2, "gesture": 1})
    check("por hora", {(x["hora"], x["tipo"]): x["n"] for x in r["por_hora"]}
          == {("2026-11-01 10:00", "face"): 2, ("2026-11-01 11:00", "gesture"): 1})
    check("por etiqueta", r["por_etiqueta"][0] == {"tipo": "face", "etiqueta": "A", "n": 2})
    check("rango por fechas", s.count(since="2026-11-01 10:30", until="2026-11-01 11:00") == 1)
    s.close()
    shutil.rmtree(d)


def test_retencion():
    print("5. Retención: borra eventos viejos")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.1)
    s.record(0, "motion", ts="2020-01-01 00:00:00.000000")
    s.record(0, "motion")
    s.flush()
    s.record(0, "gesture", ts="2020-01-01 00:00:00.000000", source="legacy")
    s.flush()
    check("borra 1 evento de 2020", s.purge(30) == 1 and s.count() == 2)
    check("conserva los heredados", s.count(type="gesture") == 1)
    s.close()
    shutil.rmtree(d)


def test_migracion(legacy_db=None):
    print("6. Migración de la tabla heredada detection_events")
    d, db = tmpdb()
    if legacy_db and os.path.exists(legacy_db):
        shutil.copy(legacy_db, db)
        origen = f"copia de {legacy_db}"
    else:
        con = sqlite3.connect(db)
        con.executescript("""CREATE TABLE detection_events (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id INTEGER,
            event_type TEXT, person_id INTEGER, confidence REAL, bounding_box TEXT, image_path TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
            INSERT INTO detection_events (camera_id, event_type, confidence, bounding_box, image_path)
            VALUES (0,'gesture',0,'{}','sessions/a.jpg'), (0,'unknown_person',0,'{}','');""")
        con.commit(); con.close()
        origen = "base sintética"
    con = sqlite3.connect(db)
    n_legacy = con.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
    ya = con.execute("SELECT name FROM sqlite_master WHERE name='events'").fetchone()
    previos = con.execute("SELECT COUNT(*) FROM events WHERE source='legacy'").fetchone()[0] if ya else 0
    con.close()
    n1 = ensure_schema(db)
    n2 = ensure_schema(db)
    s = EventStore(db)
    legacy_en_events = sqlite3.connect(db).execute(
        "SELECT COUNT(*) FROM events WHERE source='legacy'").fetchone()[0]
    nota = " (la base ya estaba migrada)" if previos else ""
    check(f"{n_legacy} eventos heredados en la tabla nueva{nota}", legacy_en_events == n_legacy
          and n1 == (0 if previos else n_legacy))
    check("no duplica al reiniciar", n2 == 0 and legacy_en_events == n_legacy)
    check("tabla heredada intacta", sqlite3.connect(db).execute(
        "SELECT COUNT(*) FROM detection_events").fetchone()[0] == n_legacy)
    s.close()
    shutil.rmtree(d)


def test_gestion_y_fotos():
    print("7. Fase 2: fotos de eventos, revisión y notas")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.1, base_dir=d)
    try:
        import numpy as np
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[100:200, 100:200] = 255
    except ImportError:
        frame = None
    e1 = s.record(0, "face", label="Desconocido", data={"bbox": [100, 100, 100, 100]}, frame=frame)
    e2 = s.record(0, "motion", value=2.0)
    s.flush(timeout=60)   # la primera foto importa OpenCV dentro del hilo escritor
    rows = s.query(newest_first=False)
    if frame is not None:
        foto = s.image_file(rows[0]["image_path"])
        check("foto del evento guardada en sessions/events/", foto is not None and os.path.getsize(foto) > 1000,
              rows[0]["image_path"])
        check("tamaño del cuadro en data", rows[0]["data"].get("frame_size") == [640, 480])
        check("frame copiado (cambiar el original no altera la foto)", e1["image_path"].startswith("sessions"))
    else:
        print("  (numpy no disponible: se omiten las pruebas de foto)")
    check("evento sin foto no tiene ruta", rows[1]["image_path"] is None)
    check("image_file rechaza rutas fuera de sessions/",
          s.image_file("../../etc/passwd") is None and s.image_file("database/test.db") is None)
    ev_id = rows[0]["id"]
    check("get por id", s.get(ev_id)["type"] == "face" and s.get(999999) is None)
    check("marcar revisado", s.update(ev_id, reviewed=True) and s.get(ev_id)["reviewed"] == 1)
    check("guardar nota", s.update(ev_id, notes="Falsa alarma") and s.get(ev_id)["notes"] == "Falsa alarma")
    check("update de id inexistente", s.update(999999, reviewed=True) is False)
    check("filtro sin revisar", s.count(reviewed=False) == 1 and s.count(reviewed=True) == 1)
    check("revisión en bloque", s.bulk_review([r["id"] for r in rows]) == 2 and s.count(reviewed=False) == 0)
    if frame is not None:
        foto = s.image_file(rows[0]["image_path"])
        con = sqlite3.connect(db)
        con.execute("UPDATE events SET ts = '2020-01-01 00:00:00'")
        con.commit(); con.close()
        s.purge(30)
        check("la retención también borra las fotos", not os.path.exists(foto))
    s.close()
    shutil.rmtree(d)


def test_reporte():
    print("8. Fase 4: reporte agregado y exportación")
    d, db = tmpdb()
    s = EventStore(db, batch_interval=0.1, base_dir=d)
    s.record(0, "motion", ts="2026-11-02 09:10:00.000000")       # lunes
    s.record(0, "motion", ts="2026-11-02 09:40:00.000000")
    s.record(0, "face", label="Desconocido", ts="2026-11-02 09:41:00.000000")
    s.record(0, "face", label="Luis", ts="2026-11-03 18:00:00.000000")   # martes
    s.record(0, "object", label="mochila", value=0.5, ts="2026-11-03 18:01:00.000000")
    s.record(0, "object", label="mochila", value=0.3, ts="2026-11-03 18:02:00.000000")
    s.record(0, "people_count", value=2, ts="2026-11-03 18:00:30.000000")
    s.record(0, "people_count", value=1, ts="2026-11-03 18:05:00.000000")
    s.record(0, "line_cross", label="down", ts="2026-11-03 18:03:00.000000")
    s.record(1, "gesture", label="manos_arriba", ts="2026-11-04 12:00:00.000000")
    s.flush()
    r = s.report("2026-11-02", "2026-11-05")
    check("total y por tipo", r["total"] == 10 and r["por_tipo"]["motion"] == 2)
    check("cubeta automática por día (3 días)", r["bucket"] == "day")
    check("serie por día", {(x["t"], x["tipo"]): x["n"] for x in r["serie"]}[("2026-11-02", "motion")] == 2)
    check("mapa de calor: lunes 9 h = 3 eventos", {(c["dow"], c["hora"]): c["n"] for c in r["calor"]}[(1, 9)] == 3)
    check("objetos con confianza media", r["objetos"] == [{"etiqueta": "mochila", "n": 2, "conf_media": 0.4}])
    check("máximo de personas", r["max_personas"] == 2)
    check("cruces", r["cruces_bajan"] == 1 and r["cruces_suben"] == 0)
    check("hora pico (martes 18 h: rostro, 2 mochilas y un cruce)", r["hora_pico"]["hora"] == "2026-11-03 18:00" and r["hora_pico"]["n"] == 4)
    r1 = s.report("2026-11-03", "2026-11-04")
    check("cubeta por hora en un día", r1["bucket"] == "hour" and r1["total"] == 6)
    check("filtro por cámara", s.report("2026-11-01", "2026-11-06", camera_id=1)["total"] == 1)
    s.update(1, reviewed=True)
    check("porcentaje revisado", s.report("2026-11-01", "2026-11-06")["pct_revisado"] == 10.0)
    check("exportación ordenada por fecha", [e["type"] for e in s.export_rows()][:2] == ["motion", "motion"])
    s.close()
    shutil.rmtree(d)


if __name__ == "__main__":
    legacy = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "pivision.db")
    test_basico()
    test_vivo_sin_robo()
    test_concurrencia_y_velocidad()
    test_resumen()
    test_retencion()
    test_migracion(legacy)
    test_gestion_y_fotos()
    test_reporte()
    print(f"\nResultado: {OK} OK, {FAIL} fallas")
    sys.exit(1 if FAIL else 0)
