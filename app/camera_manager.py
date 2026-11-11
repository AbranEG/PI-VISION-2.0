# app/camera_manager.py

import cv2, os, threading, time
from app.ai_engine import AIPipeline

# ---------------------------------------------------------------------
# Cámara individual
# ---------------------------------------------------------------------
class Camera:
    """
    Hallazgo 8: antes un solo hilo leía la cámara y corría toda la IA en serie; mientras
    se procesaba un cuadro, los siguientes se acumulaban en el búfer de la cámara y la IA
    trabajaba con imágenes atrasadas.
    Ahora hay dos hilos:
      - captura: lee sin parar y guarda solo el cuadro más reciente;
      - proceso: toma el más reciente disponible, corre el pipeline y publica el resultado.
    Los cuadros que llegan mientras se procesa se descartan (se cuentan en `stats`).
    """
    def __init__(self, device_id, width=640, height=480, fps=30):
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()

        self._latest = None          # (n, cuadro) más reciente de la cámara
        self._latest_lock = threading.Lock()
        self._new = threading.Event()
        self.stats = {"capturados": 0, "procesados": 0, "descartados": 0}
        # Apagada a mano desde el panel: no se captura, no corre la IA y no se guardan eventos
        # ni fotos. Mientras esté así, abrir el video ya no la vuelve a encender sola.
        self.apagada = False
        self._hilos = []
        self._ctl = threading.Lock()   # evita encender/apagar al mismo tiempo

        self.pipeline = AIPipeline(device_id, width, height)

    def start(self):
        with self._ctl:
            if self.running:
                return
            self._esperar_hilos()          # por si se acaba de apagar

            cap = cv2.VideoCapture(self.device_id)
            if not cap.isOpened():
                cap.release()
                raise RuntimeError(f"No se pudo abrir la cámara {self.device_id}")
            self.cap = cap
            self.apagada = False
            self.running = True
            self._hilos = [
                threading.Thread(target=self._capture_loop, name=f"captura-{self.device_id}", daemon=True),
                threading.Thread(target=self._process_loop, name=f"proceso-{self.device_id}", daemon=True),
            ]
            for t in self._hilos:
                t.start()

    def _esperar_hilos(self):
        for t in self._hilos:
            if t is not threading.current_thread():
                t.join(timeout=3)
        self._hilos = []

    def _capture_loop(self):
        cap = self.cap                     # este hilo es dueño de esta webcam abierta
        # Un archivo de video (pruebas) se lee a su propio ritmo; una cámara ya marca el ritmo sola
        es_archivo = isinstance(self.device_id, str) and os.path.isfile(self.device_id)
        periodo = 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 15.0) if es_archivo else 0.0
        try:
            self._leer(cap, periodo)
        finally:
            # Se libera aquí y no desde otro hilo: cerrar la webcam a mitad de un read() puede tronar OpenCV
            cap.release()
            if self.cap is cap:
                self.cap = None

    def _leer(self, cap, periodo):
        n = 0
        siguiente = time.time()
        while self.running and self.cap is cap:
            if periodo:
                siguiente += periodo
                time.sleep(max(0.0, siguiente - time.time()))
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            n += 1
            with self._latest_lock:
                self._latest = (n, frame)
            self.stats["capturados"] += 1
            self._new.set()

    def _process_loop(self):
        ultimo = 0
        while self.running:
            if not self._new.wait(timeout=0.5):
                continue
            self._new.clear()
            with self._latest_lock:
                item = self._latest
            if item is None or item[0] == ultimo:
                continue
            n, frame = item
            if ultimo:
                self.stats["descartados"] += max(0, n - ultimo - 1)
            ultimo = n

            frame = cv2.resize(frame, (self.width, self.height))
            processed_frame, _ = self.pipeline.process(frame)
            self.stats["procesados"] += 1

            with self.lock:
                self.frame = processed_frame

    def read_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_pipeline(self):
        return self.pipeline

    def stop(self):
        with self._ctl:
            self.running = False
            self._new.set()
            # El hilo de captura libera la webcam al salir; aquí solo se espera a que termine
            self._esperar_hilos()
            self.cap = None               # si el hilo no alcanzó a salir, lo hará al ver esto
            with self._latest_lock:
                self._latest = None
            with self.lock:
                self.frame = None

    def apagar(self):
        """Apagado manual desde el panel (libera la webcam y deja de guardar eventos)."""
        self.apagada = True
        self.stop()

    def encender(self):
        """Encendido manual; lanza RuntimeError si la webcam no responde."""
        self.start()

# ---------------------------------------------------------------------
# Manager para múltiples cámaras
# ---------------------------------------------------------------------
class CameraManager:
    def __init__(self, device_ids, width=640, height=480, fps=30):
        self.device_ids = device_ids
        self.cameras = {}
        self.width = width
        self.height = height
        self.fps = fps

        for cam_id in device_ids:
            self.cameras[cam_id] = Camera(cam_id, width, height, fps)

    def get_camera(self, cam_id):
        return self.cameras.get(cam_id)

    def get_camera_list(self):
        return list(self.cameras.keys())

    def generate(self, cam_id):
        cam = self.get_camera(cam_id)
        if not cam:
            return

        if not cam.running and not cam.apagada:
            cam.start()
            time.sleep(0.1)

        while cam.running:
            frame = cam.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            ok, jpeg = cv2.imencode('.jpg', frame)
            if not ok:
                time.sleep(0.01)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            time.sleep(1.0 / self.fps)

    @classmethod
    def detect_cameras(cls, max_test=5):
        ids = []
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ids.append(i)
                cap.release()
        return ids
