# hailo_service/detector.py
"""
Detección de objetos en el Hailo-8 con YOLOv8s (NMS dentro del chip).
Extraído del prototipo tools/hailo_probe.py (H1, paso 1), ya probado en la Pi:
11.8 ms de inferencia por frame con la webcam.
"""
import cv2
import numpy as np

COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]


# ---------------------------------------------------------------------------
# Pre y postproceso (independientes del Hailo, para poder probarlos aparte)
# ---------------------------------------------------------------------------
def letterbox(frame_bgr, size=640):
    """Redimensiona conservando proporción y rellena a size x size. Devuelve RGB uint8 y la transformación."""
    h, w = frame_bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb), (scale, px, py, w, h)


def parse_nms_by_class(raw, tf, size=640, min_score=0.35):
    """
    Salida HAILO NMS BY CLASS en FLOAT32: por clase, arreglo (k, 5) con
    [ymin, xmin, ymax, xmax, score] normalizados a la entrada de 640x640.
    Devuelve [(clase, score, x1, y1, x2, y2)] en píxeles del frame original.
    """
    scale, px, py, w, h = tf
    per_class = raw[0] if (isinstance(raw, (list, tuple)) and len(raw) == 1) else raw
    dets = []
    for cls_id, arr in enumerate(per_class):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.size == 0:
            continue
        for ymin, xmin, ymax, xmax, score in arr.reshape(-1, 5):
            if score < min_score:
                continue
            x1 = (xmin * size - px) / scale
            y1 = (ymin * size - py) / scale
            x2 = (xmax * size - px) / scale
            y2 = (ymax * size - py) / scale
            dets.append((cls_id, float(score),
                         int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2))))
    return dets


def draw(frame, dets):
    for cls_id, score, x1, y1, x2, y2 in dets:
        name = COCO[cls_id] if cls_id < len(COCO) else str(cls_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{name} {score:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


class HailoDetector:
    def __init__(self, hef_path):
        from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
        self.hef = HEF(hef_path)
        self.device = VDevice()
        cfg = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.ng = self.device.configure(self.hef, cfg)[0]
        self.ng_params = self.ng.create_params()
        in_p = InputVStreamParams.make(self.ng, format_type=FormatType.UINT8)
        out_p = OutputVStreamParams.make(self.ng, format_type=FormatType.FLOAT32)
        self.in_name = self.hef.get_input_vstream_infos()[0].name
        self.out_name = self.hef.get_output_vstream_infos()[0].name
        self._pipe = InferVStreams(self.ng, in_p, out_p)
        self._pipe.__enter__()
        self._act = self.ng.activate(self.ng_params)
        self._act.__enter__()

    def infer(self, rgb640):
        out = self._pipe.infer({self.in_name: np.expand_dims(rgb640, 0)})
        return out[self.out_name]

    def close(self):
        for ctx in (self._act, self._pipe):
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        try:
            self.device.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Varios modelos en el mismo Hailo-8 (Fase 5: objetos + pose)
# ---------------------------------------------------------------------------
class HailoRunner:
    """
    Carga uno o varios HEF en el mismo dispositivo.
    - Un modelo: activación manual (igual que el prototipo, ya probado en la Pi).
    - Varios: planificador de HailoRT (ROUND_ROBIN), que alterna el chip entre modelos;
      con el planificador no se llama a activate().
    infer(nombre, rgb640) → {nombre_de_salida: arreglo} en FLOAT32 (ya descuantizado).
    """
    def __init__(self, hefs):
        from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, InferVStreams, FormatType)
        self.multi = len(hefs) > 1
        if self.multi:
            from hailo_platform import HailoSchedulingAlgorithm
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            self.device = VDevice(params)
        else:
            self.device = VDevice()
        self.models = {}
        try:
            for name, path in hefs.items():
                hef = HEF(path)
                cfg = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
                ng = self.device.configure(hef, cfg)[0]
                in_p = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
                out_p = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
                pipe = InferVStreams(ng, in_p, out_p)
                pipe.__enter__()
                act = None
                if not self.multi:
                    act = ng.activate(ng.create_params())
                    act.__enter__()
                self.models[name] = {"path": path, "pipe": pipe, "act": act,
                                     "in": hef.get_input_vstream_infos()[0].name,
                                     "outs": [o.name for o in hef.get_output_vstream_infos()]}
        except Exception:
            self.close()
            raise

    def infer(self, name, rgb640):
        m = self.models[name]
        return m["pipe"].infer({m["in"]: np.expand_dims(rgb640, 0)})

    def close(self):
        for m in self.models.values():
            for ctx in (m["act"], m["pipe"]):
                if ctx is None:
                    continue
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
        try:
            self.device.release()
        except Exception:
            pass
