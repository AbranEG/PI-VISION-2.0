# Bitácora de desarrollo — PiVision IA (residencia 2026)

Registro cronológico de actividades, decisiones, problemas y evidencia.
Calendario: 21 sep – 18 dic 2026 · congelamiento del sistema: 22 nov.

---

## S2 · Lunes 28 de septiembre de 2026 — Reacondicionamiento del equipo

**Objetivo:** dejar una micro SD nueva con el sistema funcionando desde el repositorio.

**Actividades**
- Micro SD con Raspberry Pi OS **Debian 13 (Trixie)**, 64 bits (`aarch64`), Python del sistema 3.13.
- Se clonó `github.com/LFRC19/PiVision-IA` en `~/PiVision-IA` (último commit `7b85e78`, 21-jun-2025).
- Se revisó el código contra el ZIP de respaldo: idénticos.

**Problemas y soluciones**

| Problema | Causa | Solución |
|---|---|---|
| `mediapipe 0.10.18` y `tflite-runtime 2.14` no instalan | Solo existen para Python ≤ 3.11/3.12; Trixie trae 3.13 | Entorno `venv` con **Python 3.11** creado con `uv` (`setup_env.sh`) |
| `conda create` falla | Términos de servicio de los canales de Anaconda | Se descartó conda; se usa `uv` |
| `requirements.txt` con Django, jax, mysql, etc. | Era un `pip freeze` del sistema completo | `requirements-pi.txt` solo con lo que importa el código |
| Falta `models/mobilefacenet.tflite` | `.gitignore` excluye `*.tflite`; nunca se subió | Se obtuvo un MobileFaceNet público compatible (entrada 112×112 en [-1, 1], salida de 192 valores); sha256 `be4bc7cf…4854` |
| Credenciales en el código (Gmail, Telegram, `secret_key`) | Escritas directamente en `settings.py`, `notifier.py`, `app/__init__.py` y pruebas | Se movieron a `.env` (no versionado). **Pendiente: revocar las credenciales antiguas**, siguen en el historial público |
| `sqlite3.OperationalError: no such table: known_faces` | Tres rutas distintas a la BD (`pivision.db` en raíz vs `database/pivision.db`) | Todo apunta a `database/pivision.db`; tablas con `init_db.py` |
| Cámara índice 2 no existe | `CAMERA_IDS = [0, 2]` del equipo anterior | `CAMERA_IDS = [0]` (webcam USB HD 720P) |
| Driver del Hailo no carga | En Trixie el driver se compila con DKMS y faltaban los *headers* del kernel | `linux-headers-rpi-2712` + reinstalar `hailort-pcie-driver` + `dkms autoinstall` |

**Resultado:** `server.py` arranca y transmite en CPU. `hailortcli fw-control identify` → Hailo-8, firmware 4.23.0.

---

## S2 · Martes 29 de septiembre de 2026 — Línea base ("antes") y arquitectura

**Decisión de arquitectura — servicio de inferencia Hailo separado (opción A)**
- Las librerías del Hailo (`hailo_platform`) solo existen para Python 3.13; la app corre en 3.11.
- Se verificó que `mediapipe 1.0.1` (compatible con 3.13) **eliminó `mp.solutions`** (Holistic, FaceMesh), que usa el código.
- Migrar la app obligaría a reescribir gestos y malla facial. Se decide un **proceso aparte en Python 3.13** que atiende al Hailo; la app le envía frames.
- Ventajas: no se toca lo que funciona, saca la inferencia del hilo de captura (hallazgo 8), aísla fallos y permite medir por separado.

**Herramientas creadas**
- `tools/prepare_baseline.sh`: etiqueta `v1.0-baseline` en `7b85e78` y copia intacta del código original en `~/PiVision-baseline` (`git worktree`).
- `tools/baseline.py`: mide el pipeline real sin modificar el código (instrumentación en memoria): FPS, latencia por etapa, CPU, RAM, temperatura, *throttling*, eventos en BD. El mismo script sirve para el "después".

**Medición "antes"** (1 persona sentada frente a la cámara, 60 s tras 10 s de calentamiento)

| Métrica | CPU | `USE_TPU=true` |
|---|---|---|
| FPS de la cámara sin IA | 16.0 | 12.2 |
| FPS con el pipeline | **9.55** | **8.45** |
| Tiempo por frame, media / p95 | 103.9 / 135.5 ms | 117.7 / 128.6 ms |
| Gestos (Holistic) | 72.9 ms · 70% | 88.5 ms · 75% |
| Embedding MobileFaceNet | 21.5 ms · 20% | 21.1 ms · 16% |
| CPU del proceso | 153% | 164% |
| Temperatura media | 64.5 °C | 67.2 °C |

**Hallazgos confirmados con datos**
- **`USE_TPU` no tiene ningún efecto:** `load_tflite_model()` se llamó **0 veces**; `FaceEncoder` crea su intérprete directo en CPU. Explica por qué el equipo anterior no vio mejora.
- **Holistic es el cuello de botella** (70–75% de cada frame) y solo se usa para el gesto "manos arriba".
- **El embedding corre en ~todos los frames** (0.97/frame) aunque la malla facial corre en la mitad (0.5/frame) — hallazgo 7.
- **2 intérpretes MobileFaceNet por cámara** — hallazgo 10.
- La CPU no está saturada (~1.6 núcleos): el límite es que todo corre en serie en un hilo.
- Los FPS de la webcam dependen de la luz: el "después" debe repetir escena e iluminación.
- Limitación: sin segunda webcam no se midió el escenario de 2 cámaras.

**Benchmark del Hailo-8** (`hailortcli benchmark`)

| Modelo | FPS | Latencia |
|---|---|---|
| `yolov8s_h8.hef` (objetos, 80 clases COCO) | 309 | 6.7 ms |
| `yolov8s_pose_h8.hef` (pose) | 232 | 10.0 ms |

La pose en el Hailo tarda ~10 ms contra 73–88 ms de Holistic en CPU.

**Incidente:** tras ~80 min inactivo el Hailo dejó de responder (`Refused to change power state from D3hot to D0`). Causa: ahorro de energía de PCIe. Solución: regla udev (`power/control=on`) y `pcie_aspm=off` en `cmdline.txt`.

**H1, paso 1 — prototipo de detección con el Hailo** (`tools/hailo_probe.py`, Python 3.13 del sistema)
- Primer intento se colgó: el OpenCV del sistema abría la webcam con GStreamer. Solución: forzar el backend V4L2 y agregar un vigilante que reporta dónde se traba.
- Resultado (30 s, 455 frames, persona frente a la cámara): **15.14 FPS**. Por etapa: espera de cámara 52.6 ms, preproceso 0.9 ms, **inferencia en el Hailo 11.8 ms**, postproceso 0.2 ms.
- La IA ocupa 12.9 ms del ciclo; el límite ahora es la webcam (~15–16 FPS).
- Detección: persona en 98.2% de los frames, silla 27.0%; falsos positivos aislados (0.2%) a filtrar con umbral o por clase.

**Documento para el asesor:** "PiVision IA — Informe de avance" (Claude Docs, proyecto PI-Vision).

**Siguiente:** H1, paso 2 — convertir el prototipo en servicio con socket local y medir la latencia entre procesos.

---

## S6 · Sábado 31 de octubre de 2026 — Fase 1: capa de persistencia de eventos

**Situación:** sin avance de código entre el 29-sep y el 31-oct (solo pruebas manuales del gesto el 2-oct). Retraso de ~3 semanas respecto al calendario. Se decide recuperar siguiendo el orden del anteproyecto (F1 → F2 → F3 → F4 → F5) con fases comprimidas; los rostros en el Hailo salen del alcance (plan B).

**Qué se construyó**
- `app/event_store.py`: tabla única `events` (motion, face, people_count, line_cross, gesture, object) con índices por fecha, tipo y cámara; SQLite en modo WAL; escritura en un hilo aparte por lotes (el hilo de captura solo encola: 0.011 ms por evento); búfer en memoria con ids crecientes para el tiempo real; consultas con filtros, resumen por tipo/cámara/hora/etiqueta y retención configurable.
- Migración automática y única de los 41 eventos de `detection_events` (marcados `source='legacy'`; la tabla original queda intacta y los heredados nunca se borran).
- `app/ai_engine.py`: el pipeline registra todos los tipos de evento con reglas anti-ruido configurables en `.env` (movimiento ≥ 0.5% del cuadro y máximo uno cada 5 s; misma persona cada 10 s; conteo solo cuando cambia; cruces y gestos siempre).
- Canal en vivo `/dashboard/events/<cam>` por cursor (`Last-Event-ID`): varias pestañas ya no se roban eventos.
- API `/api/v1`: `events` (filtros por tipo, cámara, etiqueta y fechas; paginación), `events/live`, `events/summary`, `events/stats`, `cameras` (reparada: usaba un atributo inexistente). Ahora exige sesión (antes era pública).

**Hallazgos corregidos:** 2 (cámaras detenidas en cada petición), 4 (solo se guardaban gestos), 5 (búfer destructivo), 9 (`cpu_percent` bloqueaba 0.5 s).

**Pruebas**
- `tests/test_event_store.py`: 25/25 OK (persistencia, filtros, lectores simultáneos, 2,000 eventos desde 4 hilos sin pérdidas, resumen, retención, migración real de 41 eventos).
- Pipeline real con video de prueba (45 s): 35 eventos guardados — 5 rostros, 9 movimientos, 4 cambios de conteo, 17 gestos.
- Servidor Flask: API sin sesión → 401; con sesión, filtros, resumen y canal en vivo responden.

**Problema encontrado y corregido:** la retención de 30 días borraba los 22 gestos heredados del 30-sep. Se subió a 90 días y los heredados quedan excluidos.

**Hallazgo nuevo (14):** Holistic reporta "manos arriba" en un video que solo muestra un rostro (17 falsos positivos en 45 s). Defecto del detector original; se atiende en la Fase 5 al reemplazar Holistic por pose en el Hailo.

**Verificación en la Pi (31-oct):** `tests/test_event_store.py` 25/25 OK (0.012 ms por evento). `server.py` arrancó, migró los 41 eventos heredados y en el primer minuto guardó 12 eventos reales: 7 de movimiento, 2 de rostro y 3 de conteo. Ajuste: el recuadro del rostro se recorta al cuadro (la malla daba coordenadas negativas con la cara en el borde). **Fase 1 cerrada.**

**Siguiente:** Fase 2 — análisis y gestión de eventos en el dashboard (consulta, filtros, marcar como revisado, notas).

---

## S6 · Sábado 31 de octubre de 2026 (tarde) — Fase 2: análisis y gestión de eventos

**Decisiones del día**
- Foto en **todos** los eventos (no solo gestos). Configurable con `EVENT_SNAPSHOTS=all|gesture|none`.
- **Alertas por correo y Telegram desactivadas**: seguían dirigidas a la cuenta del residente anterior, que ya no participa. Se reactivan desde `/dashboard/notifications` cuando se defina el destinatario.

**Qué se construyó**
- Fotos de evento: la imagen se copia al registrar y se codifica en JPEG en el hilo escritor (no en el de captura). Ruta `sessions/events/<fecha>/cam<N>_<tipo>_<hora>.jpg`, ~29 KB cada una. La retención borra también las fotos.
- Gestión en el `EventStore`: consultar por id, marcar revisado / no revisado, notas (hasta 2,000 caracteres), revisión en bloque y filtro por estado.
- API: `GET/PATCH /api/v1/events/<id>`, `POST /api/v1/events/review`, filtro `reviewed=` en `/events`.
- Página **`/dashboard/eventos`**: contadores de 24 h por tipo y pendientes (clic = filtro), filtros por tipo, cámara, etiqueta, rango de fechas y estado, atajos (última hora, hoy, 24 h, 7 días), tarjetas con foto y recuadro del rostro, detalle con foto grande, datos técnicos, revisado y notas, navegación con teclado, revisión de la página completa, actualización cada 15 s. Funciona en celular.
- Acceso "📋 Eventos" en el dashboard. El aviso de gesto ya no dice "Correo enviado" (no siempre se envía).
- Las fotos solo se sirven desde `sessions/` y con sesión iniciada.

**Pruebas**
- `tests/test_event_store.py`: 37/37 OK (se agregaron fotos, revisión, notas, seguridad de rutas y borrado de fotos por retención).
- Pipeline real con video (30 s): 25 eventos y 14 fotos (404 KB); **16.9 FPS**, sin pérdida de rendimiento por las fotos.
- Navegador (Chromium): filtros, contadores, detalle con recuadro, guardar nota, marcar revisado, revisión en bloque y vista móvil verificados, sin errores de JavaScript.

**Pendiente de verificar en la Pi:** reiniciar `server.py`, abrir `/dashboard/eventos`, revisar fotos reales.

**Siguiente:** Fase 3 — detección de objetos con el Hailo integrada a la app (servicio + eventos `object`). Falta definir las 3 a 5 clases.

---

## S6 · Sábado 31 de octubre de 2026 (noche) — Fase 3: detección de objetos con el Hailo-8

**Verificación de la Fase 2 en la Pi:** con la cámara real, las fotos de evento se guardan en `sessions/events/` (~36 KB cada una).

**Clases elegidas (provisionales):** persona, auto, motocicleta, mochila y perro. Se cambian en `.env` (`OBJECT_CLASSES`) sin tocar código.

**Qué se construyó** (arquitectura decidida el 29-sep: servicio separado)
- `hailo_service/` (Python 3.13 del sistema): servicio que carga `yolov8s_h8.hef` una vez y atiende peticiones por un socket Unix local (`/tmp/pivision-hailo.sock`, permisos 600). Recibe el frame BGR, hace el ajuste a 640×640, la inferencia y devuelve las detecciones en píxeles del cuadro original. Varias cámaras a la vez (un hilo por conexión, acceso al Hailo con candado). Reutiliza el código del prototipo H1.
- `hailo_service/protocol.py`: mensajes `[longitud][encabezado JSON][carga binaria]`, compartido por ambos lados.
- `app/object_detector.py` (Python 3.11): cliente con reconexión automática. Si el servicio no está, devuelve `None`, avisa una vez y la app sigue sin objetos.
- Pipeline: la detección corre sobre el cuadro limpio; un evento `object` por clase como máximo cada 10 s mientras siga a la vista, con foto, confianza, cantidad y todos los recuadros. Los objetos se dibujan en rojo en el video en vivo.
- Dashboard: tarjeta "Objeto (Hailo)"; página de eventos con filtro de objetos y recuadros de todas las detecciones.
- `start_pivision.sh`: arranca el servicio Hailo y el servidor juntos; Ctrl+C detiene ambos.
- `tools/baseline.py` mide la nueva etapa `objetos_hailo`.

**Pruebas (en la nube, Hailo simulado)**
- `tests/test_hailo_service.py`: 13/13 OK con Python 3.11 y 3.13 — protocolo, filtros por clase y confianza, coordenadas, 2 cámaras simultáneas, caída y reconexión del servicio.
- Costo de la comunicación entre procesos: **1.2 ms por frame**.
- Pipeline real con el servicio aparte: 3 eventos de objeto en 25 s con foto y recuadros. Sin el servicio: la app sigue, un solo aviso, 0.01 ms por intento.

**Primera prueba en la Pi con el Hailo real (31-oct, 12:55–13:01):** el servicio arrancó y la inferencia tarda **10–11 ms** por frame. Persona detectada con 0.92–0.94 de confianza.

**Problema detectado por el usuario y corregido:**
- La persona aparecía dos veces: como "Objeto: persona" (14 eventos) y como rostro/conteo.
- El conteo por rostros parpadeaba: **41 cambios en 6 minutos** (0 → 1 → 0…) porque al voltear la cara la persona "desaparece", mientras el Hailo la veía de forma continua.
- Solución: "persona" deja de ser objeto. Los cuerpos detectados por el Hailo alimentan el **conteo de personas** y el **seguimiento/cruce de línea** (respaldo por rostros si el servicio no está). El conteo se suaviza con el valor más frecuente del último segundo (`COUNT_STABLE_S`). El seguimiento acepta 80 px de desplazamiento (cuerpos).
- Un primer intento ("el cambio debe sostenerse 1 s") falló en la prueba: con la persona perdida 1 de cada 7 cuadros nunca se registraba. Se reemplazó por la moda del último segundo.
- Prueba con Hailo simulado (persona intermitente + mochila, 30 s): 1 evento de conteo (fuente Hailo) y 4 de "mochila"; ninguno de "persona" como objeto.

**Segunda prueba del usuario (31-oct):** la mochila cuesta detectarla y puesta no se detecta; el celular no se detecta (solo el brazo como persona).
- Celular: no estaba entre las clases configuradas → se agregó `cell phone` a `OBJECT_CLASSES`.
- Mochila puesta: limitación del modelo (de frente solo se ven los tirantes). Se agregaron **umbrales por clase** (`OBJECT_MIN_SCORES`, inicial `backpack:0.35, cell phone:0.35`) para aceptar objetos pequeños o tapados sin bajar el umbral general (0.5).
- `tools/hailo_probe.py` ahora reporta la confianza por clase (máx, mediana, p25) para fijar los umbrales con datos reales.
- `tests/test_hailo_service.py`: 14/14 OK (nueva prueba de umbral por clase).

**Tercera prueba (31-oct, 13:18–13:26), cámara a mayor distancia:** con la persona a 1.5–3 m la mochila sí se detecta. Se revisaron una por una las 17 fotos de objeto (`docs/evidencia_objetos_31oct.jpg`):

| Clase | Eventos | Reales | Falsos | Confianza reales | Confianza falsos |
|---|---|---|---|---|---|
| Mochila | 4 | 4 | 0 | 0.37–0.58 | — |
| Celular | 13 | 6 | 7 | 0.54–0.95 | 0.35–0.53 |

- Los falsos "celulares" eran la mano en la cara, una oreja, un papel y un borde borroso; casi todos aparecen solo un instante.
- Ajustes: umbral de celular a **0.55** (elimina los 7 falsos y conserva 5 de 6 reales) y **persistencia**: un objeto cuenta (y se dibuja) solo si aparece en 3 de los últimos 5 cuadros (`OBJECT_PERSIST_K/N`).
- Prueba con Hailo simulado: un celular que aparece 1 de cada 4 cuadros queda descartado; una mochila constante genera sus eventos.
- Conclusión para el informe: la detección depende de la distancia de la cámara. A menos de 1 m, los objetos quedan tapados por la mano o cortados; la instalación recomendada es a 2–3 m.

**Validación del usuario:** el celular falso persiste ("cualquier cosa en la mano"). Se deja como **pendiente para la Fase 5** (hallazgo 15): confirmar que la app corría con la versión nueva (umbral 0.55 + persistencia), medir de nuevo con `hailo_probe.py` y, si sigue, evaluar quitar `cell phone` de las clases o probar un modelo más grande del Hailo Model Zoo. **Fase 3 cerrada** con esa limitación documentada.

---

## S6 · Sábado 31 de octubre de 2026 (noche) — Fase 4: reportes y estadísticas

**Qué se construyó**
- `EventStore.report()`: agregados por periodo y cámara — total y por tipo, serie por hora o por día (automático: hora si el periodo dura ≤ 2 días), mapa de calor día × hora, objetos por clase con confianza media, rostros por etiqueta, gestos, máximo de personas simultáneas por periodo, cruces de línea por dirección, hora pico y porcentaje revisado.
- API: `GET /api/v1/reports` y `GET /api/v1/events/export.csv` (mismos filtros que `/events`, hasta 50,000 filas, con BOM para que Excel respete los acentos).
- Página **`/dashboard/reportes`**: periodos (hoy, ayer, 7 días, 30 días o rango personalizado) y filtro por cámara; 7 indicadores; gráficas de actividad apilada por tipo, mapa de calor, personas por periodo y objetos por clase; tablas de rostros, gestos y cruces; "ver como tabla" para accesibilidad; tooltips; exportar CSV e **imprimir / guardar como PDF** con estilo de impresión.
- Gráficas en SVG propio, sin librerías externas (la Pi puede no tener internet). La paleta por tipo de evento se validó para contraste y daltonismo en fondo oscuro (todas las verificaciones aprobadas).
- Accesos a Reportes desde el dashboard y desde Eventos.

**Pruebas**
- `tests/test_event_store.py`: 49/49 OK (12 pruebas nuevas del reporte: cubetas, mapa de calor, confianza media, máximo de personas, cruces, hora pico, filtro por cámara, % revisado, exportación).
- Navegador (Chromium) con una semana simulada (537 eventos de prueba, solo en la copia de pruebas): gráficas, tooltips, cambio de periodo, CSV, PDF y vista móvil sin errores de JavaScript. Ejemplo: `docs/ejemplo_reporte_simulado.pdf`.

**Pendiente de verificar en la Pi:** reiniciar con `start_pivision.sh` y abrir 📊 Reportes.

**Mejora solicitada: exportación CSV clara.**
- `Exportar CSV` (`/api/v1/events/export.csv`) ahora es para usuario final: 18 columnas en español (fecha y hora separadas, cámara, tipo traducido, **Descripción** en lenguaje natural, columnas por tipo — persona identificada, cantidad de personas, % de movimiento, objeto, cantidad, confianza, dirección del cruce, gesto —, Revisado/Pendiente, notas, evidencia, origen e id). Sin JSON; lo que no aplica queda vacío.
- `Exportar datos completos` (`/api/v1/events/export_full.csv`): formato técnico anterior, con el JSON de `datos`.
- Nuevo `app/export_format.py` (solo presentación). Sin cambios en la base de datos ni en la detección.
- En los rostros, el `valor` es una distancia (no una confianza), así que no se muestra como confianza en la versión clara.
- Ambos botones en Reportes y en Eventos (en Eventos se exporta lo filtrado en pantalla).
- `tests/test_export_format.py`: 36/36 OK. Navegador: ambas descargas desde las dos páginas, mismo número de filas, sin JSON en la clara.

**Siguiente:** Fase 5 — optimización y medición del "después" (pose en el Hailo en lugar de Holistic, hallazgos 7, 8, 10, 11, 14 y 15).


---

## S6 · Sábado 31 de octubre de 2026 (noche) — Fase 5, paso 1: gestos con pose en el Hailo

**Motivo:** Holistic (MediaPipe, CPU) consumía el 70–75% de cada cuadro en el "antes" y producía gestos falsos (hallazgo 14).

**Qué se construyó**
- `hailo_service/pose.py`: decodificador de `yolov8s_pose_h8.hef` (9 tensores crudos: cajas DFL, confianza y 17 puntos COCO por escala; identificados por forma, no por nombre). Solo numpy. Gesto "manos arriba" con el mismo criterio que el original (ambas muñecas sobre la línea de hombros) pero **exigiendo visibilidad ≥ 0.5** de hombros y muñecas.
- `HailoRunner`: dos modelos (objetos + pose) en el mismo Hailo-8 con el planificador de HailoRT (ROUND_ROBIN). Si el chip no logra cargar ambos, el servicio sigue solo con objetos y lo avisa.
- Un solo viaje por cuadro: la app envía el frame una vez y recibe objetos y poses.
- Pipeline: si el servicio trae pose, los gestos salen del Hailo y **Holistic no se carga**; si no, se usa Holistic como antes. El esqueleto se dibuja en el video en vivo. El evento de gesto guarda la fuente (`hailo_pose`/`holistic`) y la caja de la persona.

**Pruebas (Hailo simulado)**
- `tests/test_pose.py`: 11/11 OK (Python 3.11 y 3.13): caja y puntos exactos, entradas como logits o probabilidades, brazos abajo, muñeca poco visible, sin personas.
- `tests/test_hailo_service.py`: 22/22 OK (pose en el mismo viaje, servicio sin pose → la app vuelve a Holistic).
- Respaldo: falla simulada al cargar dos modelos → el servicio arranca solo con objetos.
- Pipeline con video de prueba: **Holistic 0 llamadas**; 12.3 ms por cuadro sin contar el Hailo real (en la Pi se suman ~11 ms de objetos y ~10 ms de pose).
- Decodificación de pose: 0.34 ms por cuadro (en la nube).

**Verificación en la Pi (31-oct, 14:22):** `test_hailo_service.py --real` 15/15 OK — **el Hailo-8 carga objetos y pose a la vez** con el planificador. Con la app: el gesto "manos arriba" se registró una vez con fuente `hailo_pose` y el conteo salió del Hailo.

---

## S6 · Sábado 31 de octubre de 2026 (noche) — Fase 5, paso 2: correcciones de la auditoría

| Hallazgo | Corrección |
|---|---|
| 7 · embedding en cada cuadro | Solo cuando la malla facial se recalculó y, con las mismas caras, como máximo cada `FACE_REENCODE_S` (1 s); en los demás cuadros se reutiliza el resultado |
| 8 · inferencia en el hilo de captura | Dos hilos por cámara: captura (guarda solo el cuadro más reciente) y proceso (toma el más reciente; los intermedios se descartan y se cuentan) |
| 10 · intérprete duplicado | `FrameProcessor` crea encoder, matcher y BD solo si se usa `process()` (el pipeline no lo usa) |
| 11 · rostro nuevo exige reiniciar | `FaceMatcher` revisa cada 10 s si cambiaron los rostros registrados y los recarga |
| 13 · `debug=True` expuesto | Apagado por defecto; solo con `FLASK_DEBUG=true` en `.env` |

**Pruebas**
- `tests/test_auditoria_f5.py`: 6/6 OK (FrameProcessor no carga MobileFaceNet; un rostro registrado se reconoce sin reiniciar).
- Pipeline con video y pose simulada: intérpretes MobileFaceNet **2 → 1**, embeddings por cuadro **0.97 → 0.04**, Holistic **0** llamadas, 7.6 ms por cuadro sin el Hailo real; el proceso sigue el ritmo del video (15 FPS).

**Siguiente:** medición del "después" en la Pi con `tools/baseline.py`, misma escena e iluminación que el "antes".

---

## S6 · Sábado 31 de octubre de 2026 (14:28–14:31) — Medición del "después"

Misma escena que el "antes" (1 persona sentada frente a la cámara), mismo script (`tools/baseline.py`), 60 s tras 10 s de calentamiento. Archivos: `results/baseline_despues_1cam_*` y `results/baseline_despues_sin_hailo_*`.

| Métrica | Antes (original, 29-sep) | Después sin Hailo | **Después con Hailo** | Cambio |
|---|---|---|---|---|
| FPS procesados | 9.55 | 10.15 | **18.0** | **+88%** |
| Tiempo por cuadro, media | 103.9 ms | 98.4 ms | **52.9 ms** | −49% |
| Gestos (Holistic) por cuadro | 72.9 ms (70%) | 90.1 ms (92%) | **0** (pose en el Hailo) | eliminado |
| Embeddings por cuadro | 0.97 | 0.09 | **0.05** | −95% |
| Intérpretes MobileFaceNet | 2 | 1 | **1** | −50% |
| CPU del proceso (400% = 4 núcleos) | 153% | 176% | **30%** | **−80%** |
| RAM del proceso | 439 MB | 438 MB | **226 MB** | −49% |
| Temperatura media | 64.5 °C | 65.4 °C | **58.4 °C** | −6.1 °C |
| Throttling | no | no | no | — |

**Lectura**
- Las correcciones de código solas apenas cambian los FPS (9.55 → 10.15): Holistic sigue ocupando el 92% del tiempo. La mejora decisiva es pasar los gestos a la pose en el Hailo.
- Con el Hailo el sistema procesa todos los cuadros que entrega la webcam: el límite ya es la cámara, no la IA.
- De los 52.9 ms por cuadro, 44.6 ms son espera al Hailo (la CPU está libre en ese tiempo). Se esperaban ~25 ms: el chip alterna dos modelos en cada cuadro y cada cambio cuesta.

**Optimización aplicada (pendiente de medir):** pose en cada cuadro (personas, conteo, seguimiento y gestos) y objetos 1 de cada 5 cuadros (`OBJECT_DETECT_EVERY=auto`). Nueva herramienta `tools/hailo_bench.py` para medir el costo de cada combinación en el Hailo.

---

## S7 · Martes 3 de noviembre de 2026 — Rediseño de la interfaz

**Decisiones (con el usuario):** tema oscuro profesional en todo el sistema; identidad con los colores institucionales; páginas: login, monitoreo en vivo, eventos y reportes.

**Identidad visual**
- No hay paleta publicada propia del campus Tijuana; se usa la del TecNM (Manual de Identidad Gráfica 2025): **azul Pantone 294 C (#1B396A)** y grises fríos (Cool Gray 10 C). No se usan los logotipos oficiales (tienen reglas de uso); el sistema usa un logotipo propio de texto e ícono.
- Contrastes verificados: texto 14.6:1, texto secundario 9.1:1, texto tenue 5.8:1, botón principal 6.2:1.
- Colores por tipo de evento (6) validados sobre la nueva superficie (#111a2e): contraste, separación para daltonismo y diferencia visual, todos aprobados. El mismo color identifica a cada tipo en todas las páginas.

**Qué se construyó**
- `app/static/css/pivision.css`: sistema de diseño compartido (tokens, barra superior, paneles, botones, formularios, indicadores, avisos, versión para celular e impresión).
- `app/templates/base.html`: barra superior única (logotipo, En vivo / Eventos / Reportes / Notificaciones con la sección activa, estado del Hailo, usuario, salir) y pie institucional.
- **Login** nuevo: dos columnas (identidad + formulario), mostrar contraseña, aviso de acceso restringido; una columna en celular.
- **Monitoreo en vivo** rediseñado: video por cámara con FPS reales, indicador de señal y leyenda de colores; último evento de cada uno de los 6 tipos; panel de estado (CPU, memoria, disco, temperatura, acelerador, tiempos del Hailo por modelo); actividad reciente con miniaturas y "hace cuánto"; aviso al detectar un gesto.
- **Eventos y Reportes** pasados a la plantilla base sin cambiar su funcionamiento.
- API `GET /api/v1/status` (cámaras, capturados/procesados/descartados, Hailo, métricas, eventos) y temperatura del procesador en las métricas.

**Problemas encontrados en la revisión visual y corregidos:** choque de nombres de clase (`.live`, `.panel`), regla `svg { width: 100% }` de Reportes que deformaba los íconos de la barra, video que ocupaba toda la altura, campo "Etiqueta" sin estilo, botón "Siguiente" que se bajaba de línea, color del esqueleto en la leyenda (OpenCV dibuja en BGR).

**Pruebas:** navegador (Chromium) con cámara simulada y Hailo simulado: login, en vivo, eventos, detalle y reportes en escritorio y celular, sin errores de JavaScript. Capturas en `docs/diseno/`.

**Pendiente:** la página de Notificaciones aún tiene el diseño anterior.

**Botón Encender / Apagar cámara (3-nov, a petición del usuario).** Para no llenar el disco cuando no se necesita vigilar.
- Botón por cámara en Monitoreo en vivo. Apagada: se libera la webcam, no corre la IA y **no se guardan eventos ni fotos**; el video muestra "Cámara apagada" y la píldora cambia a APAGADA. Encender reabre la webcam y el video.
- `Camera.apagar()/encender()`: el hilo de captura es quien libera la webcam (cerrarla desde otro hilo a mitad de una lectura puede tronar OpenCV); al apagar se espera a que terminen ambos hilos.
- Abrir el video o la foto instantánea ya no enciende una cámara apagada por el usuario (antes cualquier petición la reiniciaba).
- API `POST /api/v1/cameras/<id>/power` con `{"on": true|false}` (requiere sesión, queda en el log quién la cambió); `/status` y `/cameras` informan `apagada`.
- Limitación: el estado vive en memoria; al reiniciar el servidor las cámaras vuelven a encender.
- Prueba en navegador: al apagar, los cuadros capturados y los eventos se quedan fijos (617 → 617; 27 → 27 en 6 s); sigue apagada tras recargar; al encender vuelve a 15 FPS; 3 clics rápidos seguidos sin fallas. Captura `docs/diseno/07_camara_apagada.png`.
