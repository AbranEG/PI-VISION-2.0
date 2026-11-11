# app/event_store.py
"""
Capa de persistencia de eventos — Fase 1.

- Una sola tabla `events` para todos los tipos (motion, face, people_count,
  line_cross, gesture y, en la Fase 3, object).
- Escritura en segundo plano: el hilo de captura solo encola; un hilo escritor
  guarda en lotes. La inferencia nunca espera al disco.
- SQLite en modo WAL: lecturas del dashboard y escrituras al mismo tiempo.
- Búfer en memoria con ids crecientes para el tiempo real: cada cliente pide
  "lo que hay después del id N", así nadie le roba eventos a nadie
  (antes, get_last_events() vaciaba la cola al leerla).

No importa OpenCV ni MediaPipe: se puede probar y usar desde cualquier script.
"""
import json
import os
import queue
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "database", "pivision.db")
SNAPSHOT_SUBDIR = os.path.join("sessions", "events")   # fotos de eventos, relativo a BASE_DIR

EVENT_TYPES = ("motion", "face", "people_count", "line_cross", "gesture", "object")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,              -- hora local ISO 'YYYY-MM-DD HH:MM:SS.ffffff'
    camera_id   INTEGER,
    type        TEXT    NOT NULL,              -- motion | face | people_count | line_cross | gesture | object
    label       TEXT,                          -- nombre, gesto, clase de objeto, dirección…
    value       REAL,                          -- confianza, conteo, % de área…
    data        TEXT,                          -- JSON con el detalle
    image_path  TEXT,
    source      TEXT    DEFAULT 'pipeline',    -- pipeline | legacy | test
    reviewed    INTEGER DEFAULT 0,             -- Fase 2: gestión de eventos
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_type_ts   ON events(type, ts);
CREATE INDEX IF NOT EXISTS idx_events_camera_ts ON events(camera_id, ts);
"""

# Mapeo de la tabla heredada detection_events → events
_LEGACY_TYPES = {"face_detected": "face", "unknown_person": "face",
                 "motion": "motion", "gesture": "gesture"}


def _now_iso():
    return datetime.now().isoformat(sep=" ", timespec="microseconds")


def connect(db_path):
    con = sqlite3.connect(db_path, timeout=10, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def ensure_schema(db_path):
    """Crea la tabla events (idempotente) y migra una sola vez los eventos heredados."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    con = connect(db_path)
    try:
        con.executescript(SCHEMA)
        has_legacy = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='detection_events'").fetchone()
        migrated = con.execute("SELECT 1 FROM events WHERE source='legacy' LIMIT 1").fetchone()
        n = 0
        if has_legacy and not migrated:
            rows = con.execute("SELECT * FROM detection_events ORDER BY id").fetchall()
            for r in rows:
                etype = _LEGACY_TYPES.get(r["event_type"], r["event_type"])
                label = "Desconocido" if r["event_type"] == "unknown_person" else None
                if r["event_type"] == "gesture":
                    label = "manos_arriba"
                con.execute(
                    "INSERT INTO events (ts, camera_id, type, label, value, data, image_path, source) "
                    "VALUES (datetime(?, 'localtime'), ?, ?, ?, ?, ?, ?, 'legacy')",
                    (r["timestamp"], r["camera_id"], etype, label, r["confidence"],
                     json.dumps({"bounding_box": r["bounding_box"], "person_id": r["person_id"],
                                 "legacy_id": r["id"]}), r["image_path"]))
                n += 1
        con.commit()
        return n
    finally:
        con.close()


class EventStore:
    def __init__(self, db_path=DEFAULT_DB_PATH, buffer_size=2000, batch_interval=0.5, retention_days=None,
                 base_dir=BASE_DIR, jpeg_quality=80):
        self.db_path = db_path
        self.base_dir = base_dir
        self.jpeg_quality = int(jpeg_quality)
        self.migrated = ensure_schema(db_path)
        self._buf = deque(maxlen=buffer_size)
        self._buf_lock = threading.Lock()
        self._seq = 0
        self._q = queue.Queue()
        self._batch_interval = batch_interval
        self._stats = {"encolados": 0, "guardados": 0, "perdidos": 0, "fotos": 0, "fotos_fallidas": 0}
        self._closed = False
        if retention_days:
            self.purge(retention_days)
        self._writer = threading.Thread(target=self._write_loop, name="EventStoreWriter", daemon=True)
        self._writer.start()

    # ------------------------------------------------------------------ escritura
    def record(self, camera_id, type, label=None, value=None, data=None, image_path=None, ts=None,
               source="pipeline", frame=None):
        """
        Registra un evento: entra al búfer en vivo y se encola para la BD. No bloquea.
        Si se pasa `frame` (imagen BGR) y no hay image_path, se guarda una foto del
        evento; la codificación JPEG ocurre en el hilo escritor, no en el de captura.
        """
        if type not in EVENT_TYPES:
            raise ValueError(f"tipo de evento desconocido: {type}")
        ts = ts or _now_iso()
        data = dict(data or {})
        snapshot = None
        if frame is not None and image_path is None:
            day, clock = ts[:10], ts[11:].replace(":", "").replace(".", "_")
            cam = "".join(ch if ch.isalnum() else "-" for ch in str(camera_id))[-20:]
            image_path = os.path.join(SNAPSHOT_SUBDIR, day, f"cam{cam}_{type}_{clock}.jpg")
            snapshot = frame.copy()
            data.setdefault("frame_size", [int(frame.shape[1]), int(frame.shape[0])])
        with self._buf_lock:
            self._seq += 1
            evt = {"id": self._seq, "timestamp": ts, "camera_id": camera_id, "type": type,
                   "label": label, "value": value, "image_path": image_path, **data}
            self._buf.append(evt)
        if not self._closed:
            self._q.put((ts, camera_id, type, label, value, json.dumps(data, ensure_ascii=False),
                         image_path, source, snapshot))
            self._stats["encolados"] += 1
        return evt

    def _write_loop(self):
        con = connect(self.db_path)
        pending, stop = [], False
        while not stop:
            try:
                item = self._q.get(timeout=self._batch_interval)
                if item is None:
                    stop = True
                else:
                    pending.append(item)
                    # junta lo que ya esté en la cola
                    while len(pending) < 500:
                        nxt = self._q.get_nowait()
                        if nxt is None:
                            stop = True
                            break
                        pending.append(nxt)
            except queue.Empty:
                pass
            if pending:
                rows = []
                for item in pending:
                    row, snap = list(item[:8]), item[8]
                    if snap is not None and not self._save_snapshot(row[6], snap):
                        row[6] = None
                    rows.append(row)
                pending = rows
                try:
                    con.executemany(
                        "INSERT INTO events (ts, camera_id, type, label, value, data, image_path, source) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pending)
                    con.commit()
                    self._stats["guardados"] += len(pending)
                except Exception as e:  # nunca tumbar la captura por un fallo de disco
                    self._stats["perdidos"] += len(pending)
                    print(f"[EventStore] Error al guardar {len(pending)} eventos: {e}")
                pending = []
        con.close()

    def _save_snapshot(self, rel_path, frame):
        try:
            import cv2   # import diferido: el módulo se puede usar sin OpenCV
            full = os.path.join(self.base_dir, rel_path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            ok = cv2.imwrite(full, frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            self._stats["fotos" if ok else "fotos_fallidas"] += 1
            return ok
        except Exception as e:
            self._stats["fotos_fallidas"] += 1
            print(f"[EventStore] No se pudo guardar la foto {rel_path}: {e}")
            return False

    def image_file(self, rel_path):
        """Ruta absoluta de una foto de evento, solo si está dentro de sessions/ (evita salir del proyecto)."""
        if not rel_path:
            return None
        root = os.path.realpath(os.path.join(self.base_dir, "sessions"))
        full = os.path.realpath(os.path.join(self.base_dir, rel_path))
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            return None
        return full

    def flush(self, timeout=15.0):
        """Espera a que la cola se vacíe (útil en pruebas y al cerrar)."""
        t_end = time.time() + timeout
        while (self._stats["guardados"] + self._stats["perdidos"] < self._stats["encolados"]
               and time.time() < t_end):
            time.sleep(0.05)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._writer.join(timeout=5)

    # ------------------------------------------------------------------ lectura en vivo
    def recent(self, since_id=0, camera_id=None, limit=200):
        """Eventos del búfer en vivo con id > since_id. No borra nada."""
        with self._buf_lock:
            items = [e for e in self._buf if e["id"] > since_id
                     and (camera_id is None or e["camera_id"] == camera_id)]
        return items[-limit:]

    # ------------------------------------------------------------------ gestión (Fase 2)
    def get(self, event_id):
        con = connect(self.db_path)
        try:
            r = con.execute("SELECT * FROM events WHERE id = ?", (int(event_id),)).fetchone()
        finally:
            con.close()
        if not r:
            return None
        d = dict(r)
        try:
            d["data"] = json.loads(d["data"]) if d["data"] else {}
        except Exception:
            pass
        return d

    def update(self, event_id, reviewed=None, notes=None):
        sets, args = [], []
        if reviewed is not None:
            sets.append("reviewed = ?"); args.append(1 if reviewed else 0)
        if notes is not None:
            sets.append("notes = ?"); args.append(str(notes)[:2000])
        if not sets:
            return False
        con = connect(self.db_path)
        try:
            n = con.execute(f"UPDATE events SET {', '.join(sets)} WHERE id = ?", args + [int(event_id)]).rowcount
            con.commit()
        finally:
            con.close()
        return n == 1

    def bulk_review(self, ids, reviewed=True):
        ids = [int(i) for i in ids][:5000]
        if not ids:
            return 0
        con = connect(self.db_path)
        try:
            n = con.execute(f"UPDATE events SET reviewed = ? WHERE id IN ({','.join('?' * len(ids))})",
                            [1 if reviewed else 0] + ids).rowcount
            con.commit()
        finally:
            con.close()
        return n

    @property
    def last_id(self):
        return self._seq

    # ------------------------------------------------------------------ consultas a la BD
    @staticmethod
    def _where(type=None, camera_id=None, label=None, since=None, until=None, reviewed=None):
        where, args = [], []
        if type:
            types = type if isinstance(type, (list, tuple)) else [type]
            where.append(f"type IN ({','.join('?' * len(types))})")
            args += list(types)
        if camera_id is not None:
            where.append("camera_id = ?"); args.append(camera_id)
        if label:
            where.append("label = ?"); args.append(label)
        if since:
            where.append("ts >= ?"); args.append(str(since))
        if until:
            where.append("ts < ?"); args.append(str(until))
        if reviewed is not None:
            where.append("reviewed = ?"); args.append(1 if reviewed else 0)
        return (" WHERE " + " AND ".join(where)) if where else "", args

    def query(self, type=None, camera_id=None, label=None, since=None, until=None,
              reviewed=None, limit=100, offset=0, newest_first=True):
        w, args = self._where(type, camera_id, label, since, until, reviewed)
        sql = "SELECT * FROM events" + w
        sql += f" ORDER BY ts {'DESC' if newest_first else 'ASC'}, id {'DESC' if newest_first else 'ASC'}"
        sql += " LIMIT ? OFFSET ?"
        args += [int(limit), int(offset)]
        con = connect(self.db_path)
        try:
            rows = con.execute(sql, args).fetchall()
        finally:
            con.close()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["data"] = json.loads(d["data"]) if d["data"] else {}
            except Exception:
                pass
            out.append(d)
        return out

    def count(self, type=None, camera_id=None, label=None, since=None, until=None, reviewed=None):
        w, args = self._where(type, camera_id, label, since, until, reviewed)
        con = connect(self.db_path)
        try:
            return con.execute("SELECT COUNT(*) FROM events" + w, args).fetchone()[0]
        finally:
            con.close()

    def summary(self, since=None, until=None, camera_id=None):
        """Conteos por tipo, por cámara y por hora (base de la Fase 4)."""
        where, args = [], []
        if since:
            where.append("ts >= ?"); args.append(str(since))
        if until:
            where.append("ts < ?"); args.append(str(until))
        if camera_id is not None:
            where.append("camera_id = ?"); args.append(camera_id)
        w = (" WHERE " + " AND ".join(where)) if where else ""
        con = connect(self.db_path)
        try:
            por_tipo = {r[0]: r[1] for r in con.execute(f"SELECT type, COUNT(*) FROM events{w} GROUP BY type", args)}
            por_camara = {str(r[0]): r[1] for r in con.execute(
                f"SELECT camera_id, COUNT(*) FROM events{w} GROUP BY camera_id", args)}
            por_hora = [{"hora": r[0], "tipo": r[1], "n": r[2]} for r in con.execute(
                f"SELECT substr(ts, 1, 13) || ':00' AS h, type, COUNT(*) FROM events{w} "
                f"GROUP BY h, type ORDER BY h", args)]
            por_etiqueta = [{"tipo": r[0], "etiqueta": r[1], "n": r[2]} for r in con.execute(
                f"SELECT type, label, COUNT(*) FROM events{w} {'AND' if w else 'WHERE'} label IS NOT NULL "
                f"GROUP BY type, label ORDER BY 3 DESC LIMIT 50", args)]
        finally:
            con.close()
        return {"total": sum(por_tipo.values()), "por_tipo": por_tipo, "por_camara": por_camara,
                "por_hora": por_hora, "por_etiqueta": por_etiqueta}

    # ------------------------------------------------------------------ reportes (Fase 4)
    def report(self, since, until, camera_id=None, bucket=None):
        """
        Agregados para la página de reportes. `bucket` = 'hour' | 'day' (por defecto: hora si
        el periodo dura 2 días o menos, si no día). Excluye nada: incluye eventos heredados.
        """
        if bucket not in ("hour", "day"):
            try:
                d0 = datetime.fromisoformat(str(since)[:19])
                d1 = datetime.fromisoformat(str(until)[:19]) if until else datetime.now()
                bucket = "hour" if (d1 - d0) <= timedelta(days=2) else "day"
            except ValueError:
                bucket = "day"
        cut = 13 if bucket == "hour" else 10
        w, args = self._where(camera_id=camera_id, since=since, until=until)
        AND = " AND " if w else " WHERE "
        con = connect(self.db_path)
        try:
            q = lambda sql, extra=(): con.execute(sql, list(args) + list(extra)).fetchall()
            total = q(f"SELECT COUNT(*) FROM events{w}")[0][0]
            revisados = q(f"SELECT COUNT(*) FROM events{w}{AND}reviewed = 1")[0][0]
            por_tipo = {t: n for t, n in q(f"SELECT type, COUNT(*) FROM events{w} GROUP BY type")}
            serie = [{"t": t, "tipo": ty, "n": n} for t, ty, n in q(
                f"SELECT substr(ts, 1, {cut}) AS b, type, COUNT(*) FROM events{w} GROUP BY b, type ORDER BY b")]
            # 0 = domingo … 6 = sábado (convención de SQLite)
            calor = [{"dow": int(d), "hora": int(h), "n": n} for d, h, n in q(
                f"SELECT strftime('%w', substr(ts, 1, 19)), CAST(substr(ts, 12, 2) AS INTEGER), COUNT(*) "
                f"FROM events{w}{AND}type != 'people_count' GROUP BY 1, 2")]
            objetos = [{"etiqueta": l, "n": n, "conf_media": round(c or 0, 2)} for l, n, c in q(
                f"SELECT label, COUNT(*), AVG(value) FROM events{w}{AND}type = 'object' GROUP BY label ORDER BY 2 DESC")]
            rostros = [{"etiqueta": l or "Desconocido", "n": n} for l, n in q(
                f"SELECT label, COUNT(*) FROM events{w}{AND}type = 'face' GROUP BY label ORDER BY 2 DESC")]
            gestos = [{"etiqueta": l or "gesto", "n": n} for l, n in q(
                f"SELECT label, COUNT(*) FROM events{w}{AND}type = 'gesture' GROUP BY label ORDER BY 2 DESC")]
            personas = [{"t": t, "max": int(m or 0)} for t, m in q(
                f"SELECT substr(ts, 1, {cut}) AS b, MAX(value) FROM events{w}{AND}type = 'people_count' "
                f"GROUP BY b ORDER BY b")]
            max_personas = max([p["max"] for p in personas] or [0])
            cruces = [{"t": t, "dir": d, "n": n} for t, d, n in q(
                f"SELECT substr(ts, 1, {cut}) AS b, label, COUNT(*) FROM events{w}{AND}type = 'line_cross' "
                f"GROUP BY b, label ORDER BY b")]
            entradas = sum(c["n"] for c in cruces if c["dir"] == "down")
            salidas = sum(c["n"] for c in cruces if c["dir"] == "up")
            pico = q(f"SELECT substr(ts, 1, 13) AS b, COUNT(*) FROM events{w}{AND}type != 'people_count' "
                     f"GROUP BY b ORDER BY 2 DESC LIMIT 1")
            por_camara = {str(c): n for c, n in q(f"SELECT camera_id, COUNT(*) FROM events{w} GROUP BY camera_id")}
        finally:
            con.close()
        return {
            "desde": str(since), "hasta": str(until) if until else None, "camara": camera_id, "bucket": bucket,
            "total": total, "revisados": revisados,
            "pct_revisado": round(100.0 * revisados / total, 1) if total else 0.0,
            "por_tipo": por_tipo, "por_camara": por_camara, "serie": serie, "calor": calor,
            "objetos": objetos, "rostros": rostros, "gestos": gestos,
            "personas": personas, "max_personas": max_personas,
            "cruces": cruces, "cruces_bajan": entradas, "cruces_suben": salidas,
            "hora_pico": {"hora": pico[0][0] + ":00", "n": pico[0][1]} if pico else None,
        }

    def export_rows(self, limit=50000, **filters):
        """Filas planas para CSV (más antiguas primero)."""
        return self.query(limit=limit, newest_first=False, **filters)

    def purge(self, retention_days):
        limit = (datetime.now() - timedelta(days=retention_days)).isoformat(sep=" ")
        con = connect(self.db_path)
        try:
            # Los eventos heredados (source='legacy') nunca se borran: son evidencia del sistema original
            fotos = [r[0] for r in con.execute(
                "SELECT image_path FROM events WHERE ts < ? AND source != 'legacy' AND image_path LIKE ?",
                (limit, SNAPSHOT_SUBDIR.replace(os.sep, "/") + "/%"))]
            n = con.execute("DELETE FROM events WHERE ts < ? AND source != 'legacy'", (limit,)).rowcount
            con.commit()
        finally:
            con.close()
        for rel in fotos:
            f = self.image_file(rel)
            if f:
                try:
                    os.remove(f)
                except OSError:
                    pass
        return n

    def stats(self):
        return dict(self._stats, en_cola=self._q.qsize(), buffer=len(self._buf), ultimo_id=self._seq)


# ---------------------------------------------------------------------- instancia compartida
_store = None
_store_lock = threading.Lock()


def get_store(db_path=None):
    """Devuelve la instancia compartida (la crea si no existe)."""
    global _store
    with _store_lock:
        if _store is None:
            retention = int(os.getenv("EVENT_RETENTION_DAYS", "90"))
            _store = EventStore(db_path or DEFAULT_DB_PATH, retention_days=retention,
                                jpeg_quality=int(os.getenv("EVENT_JPEG_QUALITY", "80")))
        return _store


def set_store(store):
    global _store
    with _store_lock:
        _store = store
