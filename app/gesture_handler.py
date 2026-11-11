# app/gesture_handler.py

import cv2, os, json, time
import mediapipe as mp
from datetime import datetime
from db import Database
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# ------------------------------------------------------
# 1. CLASE DETECTOR (actualizada con .env)
# ------------------------------------------------------
class GestureDetector:
    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        # Leer confianza desde .env o usar 0.7 por defecto
        import os
        det_conf = float(os.getenv("GESTURE_DETECTION_CONFIDENCE", 0.7))

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=det_conf,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb)

        h, w = frame.shape[:2]
        output = {'pose': [], 'left_hand': [], 'right_hand': [], 'gestures': []}

        # Pose
        if results.pose_landmarks:
            output['pose'] = [
                (int(lm.x * w), int(lm.y * h))
                for lm in results.pose_landmarks.landmark
            ]

        # Manos
        if results.left_hand_landmarks:
            output['left_hand'] = [
                (int(lm.x * w), int(lm.y * h))
                for lm in results.left_hand_landmarks.landmark
            ]
        if results.right_hand_landmarks:
            output['right_hand'] = [
                (int(lm.x * w), int(lm.y * h))
                for lm in results.right_hand_landmarks.landmark
            ]

        # Gesto "manos_arriba"
        pose = output['pose']
        if pose and len(pose) > 16:
            shoulder_y = (pose[11][1] + pose[12][1]) // 2
            if pose[15][1] < shoulder_y and pose[16][1] < shoulder_y:
                output['gestures'].append('manos_arriba')

        return output

# ------------------------------------------------------
# 2. CLASE HANDLER (sin cambios)
# ------------------------------------------------------
class GestureHandler:
    def __init__(self, session_dir, cooldown=1.0, log_to_db=True):
        self.detector    = GestureDetector()
        self.log_to_db   = log_to_db   # False cuando el pipeline registra en el EventStore
        self.db          = Database() if log_to_db else None
        self.session_dir = session_dir
        self.cooldown    = cooldown
        self.last_event  = 0.0
        os.makedirs(self.session_dir, exist_ok=True)

    def analyze(self, cam_id, frame):
        """
        Devuelve el nombre del gesto detectado o None.
        """
        results = self.detector.process(frame)
        if not results['gestures']:
            return None

        gesture = results['gestures'][0]
        now = time.time()
        if now - self.last_event < self.cooldown:
            return None

        self.last_event = now

        ts = int(now)
        filename = os.path.join(
            self.session_dir, f"cam{cam_id}_{gesture}_{ts}.jpg")
        cv2.imwrite(filename, frame)

        if self.log_to_db:
          self.db.log_event(
            camera_id    = cam_id,
            event_type   = 'gesture',
            person_id    = None,
            confidence   = 0.0,
            bounding_box = json.dumps({
                'x': 0, 'y': 0,
                'w': frame.shape[1],
                'h': frame.shape[0]
            }),
            image_path   = filename
        )

        print(f"[GESTO] Cam {cam_id}: {gesture} detectado -> {filename}")
        return (gesture, filename)
