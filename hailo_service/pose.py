# hailo_service/pose.py
"""
Decodificación de YOLOv8-pose (yolov8s_pose_h8.hef) para el Hailo-8 — Fase 5.

El HEF entrega 9 tensores crudos, 3 por escala (stride 32, 16 y 8 → 20x20, 40x40, 80x80):
    caja    (H, W, 64)  distribución DFL: 4 lados × 16 bins
    score   (H, W, 1)   confianza de "persona"
    puntos  (H, W, 51)  17 puntos COCO × (x, y, visibilidad)

Se identifican por forma (canales y tamaño), no por nombre, para no depender
de los nombres internos de las capas. Fórmulas equivalentes a las de Ultralytics
y del Hailo Model Zoo:
    lados   = Σ softmax(bins) · [0..15] · stride
    caja    = centro ± lados               (centro = (celda + 0.5) · stride)
    punto   = stride · (2·raw − 0.5) + centro
Solo usa numpy: se prueba sin el Hailo.
"""
import numpy as np

KEYPOINTS = ["nariz", "ojo_izq", "ojo_der", "oreja_izq", "oreja_der", "hombro_izq", "hombro_der",
             "codo_izq", "codo_der", "muñeca_izq", "muñeca_der", "cadera_izq", "cadera_der",
             "rodilla_izq", "rodilla_der", "tobillo_izq", "tobillo_der"]
# Esqueleto para dibujar (pares de índices COCO)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]
REG_MAX = 16


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _as_prob(x):
    """Si los valores ya están en [0, 1] se dejan; si son logits se pasa por sigmoide."""
    return x if (x.size == 0 or (x.min() >= 0.0 and x.max() <= 1.0)) else _sigmoid(x)


def _group_by_scale(outputs):
    """{nombre: tensor} → {H: {"box":…, "score":…, "kpts":…}} usando la forma de cada tensor."""
    scales = {}
    for arr in outputs.values():
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 4:
            a = a[0]
        if a.ndim != 3:
            continue
        h, w, c = a.shape
        kind = {4 * REG_MAX: "box", 1: "score", 51: "kpts"}.get(c)
        if kind:
            scales.setdefault(h, {})[kind] = a
    return {h: s for h, s in scales.items() if len(s) == 3}


def _nms(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_i + area_r - inter, 1e-6)
        order = rest[iou < iou_thr]
    return keep


def decode_pose(outputs, tf, size=640, score_thr=0.5, iou_thr=0.65, max_det=10):
    """
    outputs: dict de tensores del HEF; tf: (scale, px, py, w, h) de letterbox().
    Devuelve [{"score", "bbox": [x, y, w, h], "keypoints": [[x, y, vis], …17]}] en píxeles del cuadro original.
    """
    scale, px, py, W0, H0 = tf
    all_boxes, all_scores, all_kpts = [], [], []
    for h, t in sorted(_group_by_scale(outputs).items()):
        stride = size / h
        w = t["score"].shape[1]
        sc = _as_prob(t["score"].reshape(-1))
        idx = np.nonzero(sc >= score_thr)[0]
        if idx.size == 0:
            continue
        gy, gx = np.divmod(idx, w)
        cx, cy = (gx + 0.5) * stride, (gy + 0.5) * stride
        # Caja: distribución DFL → distancia esperada a cada lado
        d = t["box"].reshape(-1, 4, REG_MAX)[idx]
        d = np.exp(d - d.max(axis=-1, keepdims=True))
        d = d / d.sum(axis=-1, keepdims=True)
        dist = (d * np.arange(REG_MAX, dtype=np.float32)).sum(-1) * stride   # (n, 4): izq, arriba, der, abajo
        boxes = np.stack([cx - dist[:, 0], cy - dist[:, 1], cx + dist[:, 2], cy + dist[:, 3]], axis=1)
        # Puntos del cuerpo
        kall = t["kpts"].reshape(-1, 17, 3)
        vis = _as_prob(kall[..., 2])[idx]
        k = kall[idx]
        kx = stride * (2.0 * k[..., 0] - 0.5) + cx[:, None]
        ky = stride * (2.0 * k[..., 1] - 0.5) + cy[:, None]
        all_boxes.append(boxes); all_scores.append(sc[idx]); all_kpts.append(np.stack([kx, ky, vis], -1))
    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes); scores = np.concatenate(all_scores); kpts = np.concatenate(all_kpts)
    keep = _nms(boxes, scores, iou_thr)[:max_det]
    out = []
    for i in keep:
        x1, y1, x2, y2 = (boxes[i] - [px, py, px, py]) / scale
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(W0 - 1.0, x2), min(H0 - 1.0, y2)
        kp = [[round(float((x - px) / scale), 1), round(float((y - py) / scale), 1), round(float(v), 3)]
              for x, y, v in kpts[i]]
        out.append({"score": round(float(scores[i]), 3),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], "keypoints": kp})
    return out


# ----------------------------------------------------------------------------- gestos
def manos_arriba(pose, min_vis=0.5):
    """
    Mismo criterio que el detector original con Holistic (ambas muñecas por encima de la
    línea de hombros), pero exigiendo que hombros y muñecas sean visibles: Holistic
    "inventaba" brazos y producía falsos positivos (hallazgo 14).
    """
    kp = pose["keypoints"]
    hi, hd, mi, md = kp[5], kp[6], kp[9], kp[10]
    if min(hi[2], hd[2], mi[2], md[2]) < min_vis:
        return False
    hombros_y = (hi[1] + hd[1]) / 2.0
    return mi[1] < hombros_y and md[1] < hombros_y


GESTOS = {"manos_arriba": manos_arriba}


def gestos(poses, min_vis=0.5):
    """Lista de gestos detectados en cualquiera de las personas."""
    out = []
    for p in poses or []:
        for nombre, fn in GESTOS.items():
            if fn(p, min_vis) and nombre not in out:
                out.append(nombre)
    return out
