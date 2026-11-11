# Baseline PiVision IA — despues_sin_hailo

- Fecha: 2026-10-31T14:30:07  |  Código: `v1.0-baseline-dirty`  |  Escena: 1 persona sentada frente a cámara
- Equipo: Raspberry Pi 5 Model B Rev 1.1 · Debian GNU/Linux 13 (trixie) · Python 3.11.16
- Cámaras: 0 · Resolución del pipeline 640x480 · 60s medidos tras 10s de warmup
- Variables: USE_TPU=false, DETECTION_FRAME_SKIP=2, FACE_MATCH_THRESHOLD=0.45, GESTURE_DETECTION_CONFIDENCE=0.7

## FPS

| Cámara | Sin IA | Con pipeline |
|---|---|---|
| 0 | 12.6 | 10.15 |

FPS total del sistema: **10.15**

## Latencia por etapa

| Etapa | Llamadas/frame | Media ms | p50 | p95 | Máx | % del pipeline |
|---|---|---|---|---|---|---|
| movimiento | 1.0 | 0.77 | 0.74 | 0.97 | 1.54 | 0.8 |
| malla_facial | 1.0 | 4.83 | 9.21 | 10.01 | 11.19 | 4.9 |
| normalizacion | 0.09 | 4.23 | 4.2 | 4.58 | 5.33 | 0.4 |
| embedding | 0.09 | 21.18 | 20.99 | 21.58 | 26.77 | 1.9 |
| comparacion | 0.09 | 0.04 | 0.0 | 0.41 | 0.49 | 0.0 |
| gestos | 1.0 | 90.12 | 90.24 | 92.85 | 103.67 | 91.6 |
| tracking | 1.0 | 0.11 | 0.11 | 0.12 | 0.44 | 0.1 |
| objetos_hailo | 0 | — | — | — | — | — |
| mp_facemesh_inferencia | 0.5 | 9.05 | 8.98 | 9.58 | 10.51 | 4.6 |
| mp_holistic_inferencia | 1.0 | 89.79 | 89.94 | 92.52 | 103.35 | 91.3 |
| pipeline_total | 1.0 | 98.4 | 94.11 | 126.06 | 142.94 | 100.0 |

## Recursos

| Métrica | Media | Máx |
|---|---|---|
| CPU total % | 44.5 | 51.4 |
| CPU del proceso % (400 = 4 núcleos) | 176.2 | 181.9 |
| RAM del proceso MB | 437.5 | 438.3 |
| RAM del sistema % | 25.4 | 26.5 |
| Temperatura °C | 65.4 | 68.8 |
| Frecuencia CPU MHz | 2391.7 | 2400.0 |

- Núcleo más cargado (media): 47.7 %
- RAM usada al cargar modelos: 30.6 MB
- Throttling: inicio `throttled=0x0` · fin `throttled=0x0`

## Evidencia de hallazgos

- Intérpretes MobileFaceNet creados: **1** para 1 cámara(s)
- `load_tflite_model()` (ruta con acelerador / USE_TPU) llamada: **0** veces
- Inferencias reales de malla facial por frame: 0.5 (DETECTION_FRAME_SKIP=2) · embeddings por frame: 0.09
- Eventos guardados en BD durante la prueba: {'face': 7, 'motion': 12, 'people_count': 1}
- Rostros conocidos en BD: 0
- Notificaciones suprimidas: 0
