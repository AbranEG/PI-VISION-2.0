# face_matcher.py

import numpy as np
import json
import sqlite3
import time
from db import Database

# Cada cuánto se revisa si cambiaron los rostros registrados (consulta muy ligera)
RELOAD_CHECK_S = 10.0

class FaceMatcher:
    def __init__(self, db: Database, threshold: float = 0.6):
        self.db = db
        self.threshold = threshold
        self.known_faces = self._load_known_faces()
        self._firma = self._firma_bd()
        self._next_check = time.time() + RELOAD_CHECK_S

    def _firma_bd(self):
        """(cantidad, id máximo) de known_faces: si cambia, hay que recargar."""
        try:
            con = sqlite3.connect(self.db.db_path)
            try:
                return tuple(con.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM known_faces").fetchone())
            finally:
                con.close()
        except Exception:
            return None

    def maybe_reload(self, force=False):
        """
        Hallazgo 11: antes los rostros se cargaban una sola vez y registrar uno nuevo
        obligaba a reiniciar. Ahora se recargan solos (revisión cada RELOAD_CHECK_S).
        """
        now = time.time()
        if not force and now < self._next_check:
            return False
        self._next_check = now + RELOAD_CHECK_S
        firma = self._firma_bd()
        if force or firma != self._firma:
            self.known_faces = self._load_known_faces()
            self._firma = firma
            print(f"[FaceMatcher] Rostros conocidos recargados: {len(self.known_faces)}")
            return True
        return False

    def _load_known_faces(self):
        known = []
        for row in self.db.get_known_faces():
            try:
                vector = np.array(json.loads(row['face_encoding']))
                known.append({
                    'id': row['id'],
                    'name': row['name'],
                    'encoding': vector
                })
            except Exception as e:
                print(f"[!] Error al cargar encoding de {row['name']}: {e}")
        return known

    def match(self, new_encoding: np.ndarray):
        self.maybe_reload()
        if not self.known_faces:
            return None  # No hay caras registradas aún

        best_match = None
        min_distance = float('inf')

        for face in self.known_faces:
            distance = np.linalg.norm(new_encoding - face['encoding'])
            if distance < min_distance:
                min_distance = distance
                best_match = face

        if min_distance <= self.threshold:
            return {
                'id': best_match['id'],
                'name': best_match['name'],
                'distance': round(min_distance, 4)
            }
        else:
            return None
