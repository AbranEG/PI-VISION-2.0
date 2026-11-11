import cv2
import numpy as np
import json

from face_encoder import FaceEncoder
from face_matcher import FaceMatcher
from db import Database

class FrameProcessor:
    def __init__(self, threshold: float = 0.6):
        # Para detección de movimiento
        self.previous_frame = None
        self.threshold = threshold
        # Hallazgo 10: el pipeline solo usa detect_motion(); el encoder, el matcher y la BD
        # se crean únicamente si alguien llama a process() (antes: un intérprete TFLite
        # extra por cámara que nunca se usaba).
        self._encoder = self._matcher = self._db = None

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = FaceEncoder()
        return self._encoder

    @property
    def db(self):
        if self._db is None:
            self._db = Database()
        return self._db

    @property
    def matcher(self):
        if self._matcher is None:
            self._matcher = FaceMatcher(self.db, threshold=self.threshold)
        return self._matcher

    def preprocess(self, frame, blur_type="gaussian", kernel_size=(5, 5)):
        """
        Convierte a escala de grises y aplica desenfoque para reducir ruido.
        blur_type: 'gaussian' o 'median'
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if blur_type == "gaussian":
            gray = cv2.GaussianBlur(gray, kernel_size, 0)
        elif blur_type == "median":
            gray = cv2.medianBlur(gray, kernel_size[0])  # sólo necesita un valor
        return gray

    def detect_motion(self, frame):
        """
        Compara el frame actual con el anterior y detecta regiones con movimiento.
        Devuelve el umbral binarizado y una lista de contornos.
        """
        gray = self.preprocess(frame)

        if self.previous_frame is None:
            self.previous_frame = gray
            return None, []

        diff = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        self.previous_frame = gray

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return thresh, contours

    def process(self, frame):
        """
        1) Detecta movimiento.
        2) Para cada región detectada, recorta, codifica, compara y loggea.
        3) Anota en el frame el nombre o 'Desconocido'.
        Devuelve el frame anotado, el threshold y los contornos para visualización.
        """
        thresh, contours = self.detect_motion(frame)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Evitar regiones muy pequeñas
            if w < 50 or h < 50:
                continue

            # 2.1) Recortar cara (o región de interés)
            face_img = frame[y:y+h, x:x+w]

            # 2.2) Encoding facial
            embedding = self.encoder.encode(face_img)

            # 2.3) Matching con la base de datos
            match = self.matcher.match(embedding)
            if match:
                event_type = 'face_detected'
                person_id  = match['id']
                label      = match['name']
                confidence = match['distance']
            else:
                event_type = 'unknown_person'
                person_id  = None
                label      = 'Desconocido'
                confidence = 0.0

            # 2.4) Log en la tabla detection_events
            self.db.log_event(
                camera_id    = 1,  # ← reemplaza con tu ID de cámara real
                event_type   = event_type,
                person_id    = person_id,
                confidence   = confidence,
                bounding_box = json.dumps({'x': x, 'y': y, 'w': w, 'h': h}),
                image_path   = ''  # ← opcional: guarda la ruta de un recorte o captura
            )

            # 2.5) Anotar el frame con el nombre
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, thresh, contours
