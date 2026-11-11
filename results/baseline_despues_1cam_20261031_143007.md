# Baseline PiVision IA — despues_1cam

- Fecha: 2026-10-31T14:28:49  |  Código: `v1.0-baseline-dirty`  |  Escena: 1 persona sentada frente a cámara
- Equipo: Raspberry Pi 5 Model B Rev 1.1 · Debian GNU/Linux 13 (trixie) · Python 3.11.16
- Cámaras: 0 · Resolución del pipeline 640x480 · 60s medidos tras 10s de warmup
- Variables: USE_TPU=false, DETECTION_FRAME_SKIP=2, FACE_MATCH_THRESHOLD=0.45, GESTURE_DETECTION_CONFIDENCE=0.7

## FPS

| Cámara | Sin IA | Con pipeline |
|---|---|---|
| 0 | 14.4 | 18.0 |

FPS total del sistema: **18.0**

## Latencia por etapa

| Etapa | Llamadas/frame | Media ms | p50 | p95 | Máx | % del pipeline |
|---|---|---|---|---|---|---|
| movimiento | 1.0 | 1.03 | 0.97 | 1.47 | 1.89 | 1.9 |
| malla_facial | 1.0 | 5.16 | 4.68 | 12.46 | 15.46 | 9.7 |
| normalizacion | 0.05 | 4.56 | 4.25 | 5.85 | 6.28 | 0.4 |
| embedding | 0.05 | 23.53 | 21.42 | 28.62 | 28.94 | 2.3 |
| comparacion | 0.05 | 0.04 | 0.0 | 0.4 | 0.5 | 0.0 |
| gestos | 0 | — | — | — | — | — |
| tracking | 1.0 | 0.13 | 0.12 | 0.16 | 0.33 | 0.2 |
| objetos_hailo | 1.0 | 44.64 | 44.32 | 46.25 | 47.47 | 84.3 |
| mp_facemesh_inferencia | 0.5 | 9.64 | 9.17 | 11.94 | 14.62 | 9.1 |
| mp_holistic_inferencia | 0 | — | — | — | — | — |
| pipeline_total | 1.0 | 52.94 | 51.87 | 80.26 | 95.75 | 100.0 |

## Recursos

| Métrica | Media | Máx |
|---|---|---|
| CPU total % | 15.5 | 18.4 |
| CPU del proceso % (400 = 4 núcleos) | 30.0 | 36.0 |
| RAM del proceso MB | 225.8 | 226.5 |
| RAM del sistema % | 23.1 | 23.3 |
| Temperatura °C | 58.4 | 59.5 |
| Frecuencia CPU MHz | 2251.7 | 2400.0 |

- Núcleo más cargado (media): 16.1 %
- RAM usada al cargar modelos: 31.7 MB
- Throttling: inicio `throttled=0x0` · fin `throttled=0x0`

## Evidencia de hallazgos

- Intérpretes MobileFaceNet creados: **1** para 1 cámara(s)
- `load_tflite_model()` (ruta con acelerador / USE_TPU) llamada: **0** veces
- Inferencias reales de malla facial por frame: 0.5 (DETECTION_FRAME_SKIP=2) · embeddings por frame: 0.05
- Eventos guardados en BD durante la prueba: {'face': 7, 'gesture': 1, 'line_cross': 1, 'motion': 13, 'object': 2, 'people_count': 1}
- Rostros conocidos en BD: 0
- Notificaciones suprimidas: 1
