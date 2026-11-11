#!/usr/bin/python3
"""
hailo_probe.py — H1, paso 1: detección de objetos en el Hailo-8 desde Python.

Corre con el Python 3.13 DEL SISTEMA (donde está hailo_platform), no con el venv:
    /usr/bin/python3 tools/hailo_probe.py                 # webcam 0, 30 s
    /usr/bin/python3 tools/hailo_probe.py --source foto.jpg
    /usr/bin/python3 tools/hailo_probe.py --seconds 60 --save-every 50

Mide por etapa (captura, preproceso, inferencia Hailo, postproceso), cuenta
detecciones por clase y guarda cuadros anotados en results/hailo_probe/.
Detén server.py antes: la webcam solo la puede abrir un proceso a la vez.
"""
import argparse
import faulthandler
import json
import os
import sys
import time
from collections import Counter, defaultdict

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


def stats(v):
    a = np.array(v) if v else np.array([0.0])
    return {"n": len(v), "media_ms": round(float(a.mean()), 2), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2), "max_ms": round(float(a.max()), 2)}


# ---------------------------------------------------------------------------
# Hailo
# ---------------------------------------------------------------------------
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="/usr/share/hailo-models/yolov8s_h8.hef")
    ap.add_argument("--source", default="0", help="índice de webcam o ruta de imagen/video")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--save-every", type=int, default=100, help="guardar 1 de cada N cuadros anotados")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "results", "hailo_probe"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import hailo_platform
    print(f"[probe] Python {sys.version.split()[0]} · hailo_platform {hailo_platform.__version__} · "
          f"cv2 {cv2.__version__} · numpy {np.__version__}")
    print(f"[probe] HEF: {a.hef}")

    faulthandler.dump_traceback_later(20, exit=True)
    det = HailoDetector(a.hef)
    faulthandler.cancel_dump_traceback_later()
    print("[probe] Hailo configurado", flush=True)
    t = {"captura": [], "preproceso": [], "inferencia_hailo": [], "postproceso": [], "total": []}
    classes = Counter()
    frames_with = defaultdict(int)
    scores = defaultdict(list)      # confianza de cada detección por clase
    n_frames = 0
    stamp = time.strftime("%Y%m%d_%H%M%S")

    try:
        src = a.source
        is_image = os.path.isfile(src) and src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        if is_image:
            cap = None
        elif src.isdigit():
            cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)   # evita GStreamer, que se traba con webcams USB
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            cap = cv2.VideoCapture(src)
        if cap is not None and not cap.isOpened():
            sys.exit(f"[probe] No se pudo abrir la fuente {src} (¿server.py sigue corriendo?)")
        still = cv2.imread(src) if is_image else None

        # Pruebas por separado: si algo se traba >15 s, imprime dónde y sale
        faulthandler.dump_traceback_later(15, exit=True)
        t0 = time.perf_counter()
        if still is not None:
            ok, frame0 = True, still.copy()
        else:
            ok, frame0 = cap.read()
        if not ok:
            sys.exit("[probe] La cámara abrió pero no entrega frames")
        print(f"[probe] cámara OK: frame {frame0.shape[1]}x{frame0.shape[0]} en {(time.perf_counter()-t0)*1000:.0f} ms"
              f" (backend {cap.getBackendName() if cap is not None else 'imagen'})", flush=True)
        t0 = time.perf_counter()
        det.infer(letterbox(frame0)[0])
        print(f"[probe] Hailo OK: primera inferencia en {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
        faulthandler.cancel_dump_traceback_later()

        t_start = time.perf_counter()
        t_measure = t_start + a.warmup
        t_end = t_measure + a.seconds
        print(f"[probe] warmup {a.warmup}s, midiendo {a.seconds}s …")
        while True:
            now = time.perf_counter()
            if now >= t_end:
                break
            faulthandler.dump_traceback_later(15, exit=True)
            c0 = time.perf_counter()
            if still is not None:
                ok, frame = True, still.copy()
            else:
                ok, frame = cap.read()
            if not ok:
                if cap is not None and not str(src).isdigit():
                    break
                continue
            c1 = time.perf_counter()
            rgb, tf = letterbox(frame)
            c2 = time.perf_counter()
            raw = det.infer(rgb)
            c3 = time.perf_counter()
            dets = parse_nms_by_class(raw, tf, min_score=a.min_score)
            c4 = time.perf_counter()

            if c0 < t_measure:
                continue
            n_frames += 1
            t["captura"].append((c1 - c0) * 1000)
            t["preproceso"].append((c2 - c1) * 1000)
            t["inferencia_hailo"].append((c3 - c2) * 1000)
            t["postproceso"].append((c4 - c3) * 1000)
            t["total"].append((c4 - c0) * 1000)
            seen = set()
            for d in dets:
                classes[COCO[d[0]]] += 1
                scores[COCO[d[0]]].append(d[1])
                seen.add(COCO[d[0]])
            for s in seen:
                frames_with[s] += 1
            if n_frames % a.save_every == 1:
                cv2.imwrite(os.path.join(a.out, f"{stamp}_f{n_frames:05d}.jpg"), draw(frame.copy(), dets))
        faulthandler.cancel_dump_traceback_later()
        elapsed = time.perf_counter() - t_measure
    finally:
        det.close()
        if 'cap' in locals() and cap is not None:
            cap.release()

    fps = n_frames / elapsed if elapsed > 0 else 0
    result = {
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hef": a.hef, "fuente": a.source, "segundos": round(elapsed, 1), "frames": n_frames,
        "fps": round(fps, 2),
        "etapas": {k: stats(v) for k, v in t.items()},
        "clases_detectadas": dict(classes.most_common()),
        "frames_con_clase_pct": {k: round(100 * v / max(1, n_frames), 1) for k, v in frames_with.items()},
        "confianza_por_clase": {k: {"max": round(max(v), 2), "mediana": round(float(np.median(v)), 2),
                                    "p25": round(float(np.percentile(v, 25)), 2), "n": len(v)}
                                for k, v in scores.items()},
        "python": sys.version.split()[0], "hailo_platform": hailo_platform.__version__,
    }
    path = os.path.join(a.out, f"hailo_probe_{stamp}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n## Hailo probe — {os.path.basename(a.hef)}\n")
    print(f"Frames: {n_frames} en {elapsed:.1f}s → **{fps:.2f} FPS**\n")
    print("| Etapa | Media ms | p50 | p95 | Máx |\n|---|---|---|---|---|")
    for k, s in result["etapas"].items():
        print(f"| {k} | {s['media_ms']} | {s['p50_ms']} | {s['p95_ms']} | {s['max_ms']} |")
    print("\n| Clase | % de frames | Confianza máx | Mediana | p25 |\n|---|---|---|---|---|")
    for k, v in sorted(result["frames_con_clase_pct"].items(), key=lambda x: -x[1]):
        c = result["confianza_por_clase"][k]
        print(f"| {k} | {v}% | {c['max']} | {c['mediana']} | {c['p25']} |")
    if not result["frames_con_clase_pct"]:
        print("| (ninguna) | | | | |")
    print(f"\nResultados: {path}")


if __name__ == "__main__":
    main()
