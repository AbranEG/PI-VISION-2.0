#!/usr/bin/env python3
"""
Pruebas de la exportación CSV clara (app/export_format.py).
Uso:  python tests/test_export_format.py
Solo prueba presentación: no usa base de datos, cámara ni modelos.
"""
import importlib.util
import os
import sys

_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "export_format.py")
_spec = importlib.util.spec_from_file_location("export_format", _p)
ef = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ef)

OK, FAIL = 0, 0


def check(name, cond, detail=""):
    global OK, FAIL
    if cond:
        OK += 1
        print(f"  [OK]   {name}")
    else:
        FAIL += 1
        print(f"  [FALLA] {name} → {detail}")


def ev(**kw):
    base = {"id": 1, "ts": "2026-10-31 13:22:55.123456", "camera_id": 0, "type": "motion", "label": None,
            "value": None, "data": {}, "image_path": None, "source": "pipeline", "reviewed": 0, "notes": None}
    base.update(kw)
    return base


def solo(fila, *llenas):
    """Las columnas específicas de tipo que no están en `llenas` deben ir vacías."""
    especificas = ["Persona identificada", "Cantidad de personas", "Movimiento (% del cuadro)", "Objeto",
                   "Cantidad de objetos", "Confianza (%)", "Dirección del cruce", "Gesto"]
    return all(fila[c] == "" for c in especificas if c not in llenas)


print("1. Columnas comunes")
f = ef.fila_clara(ev(reviewed=1, notes="Falsa alarma", image_path="sessions/events/2026-10-31/a.jpg"))
check("columnas en orden y en español", list(f.keys()) == ef.COLUMNAS)
check("fecha y hora separadas", f["Fecha"] == "31/10/2026" and f["Hora"] == "13:22:55", (f["Fecha"], f["Hora"]))
check("cámara legible", f["Cámara"] == "Cámara 0")
check("estado Revisado", f["Estado"] == "Revisado")
check("estado Pendiente", ef.fila_clara(ev())["Estado"] == "Pendiente")
check("notas y evidencia", f["Notas"] == "Falsa alarma" and f["Evidencia (imagen)"].endswith("a.jpg"))
check("origen e id", f["Origen"] == "Sistema actual" and f["ID del evento"] == 1)
check("ningún valor es JSON", not any(isinstance(v, (dict, list)) or str(v).startswith("{") for v in f.values()))

print("2. Movimiento")
f = ef.fila_clara(ev(type="motion", value=2.3, data={"n": 12, "area_pct": 2.3}))
check("descripción", f["Descripción"] == "Movimiento detectado en 2.3% del cuadro", f["Descripción"])
check("porcentaje", f["Movimiento (% del cuadro)"] == "2.3")
check("tipo traducido", f["Tipo de evento"] == "Movimiento")
check("otras columnas vacías", solo(f, "Movimiento (% del cuadro)"))

print("3. Rostro")
f = ef.fila_clara(ev(type="face", label="Desconocido", data={"match": False, "bbox": [1, 2, 3, 4]}))
check("desconocido", f["Descripción"] == "Rostro desconocido detectado" and f["Persona identificada"] == "Desconocido")
f = ef.fila_clara(ev(type="face", label="Luis", value=0.31, data={"match": True}))
check("reconocido", f["Descripción"] == "Rostro reconocido: Luis", f["Descripción"])
check("la distancia del rostro no se presenta como confianza", f["Confianza (%)"] == "")
check("otras columnas vacías", solo(f, "Persona identificada"))

print("4. Personas")
f = ef.fila_clara(ev(type="people_count", value=2, data={"count": 2, "fuente": "hailo", "boxes": [[1, 2, 3, 4]]}))
check("descripción con fuente", f["Descripción"] == "2 personas en cuadro (detección por cuerpo)", f["Descripción"])
check("cantidad", f["Cantidad de personas"] == 2)
check("singular", ef.fila_clara(ev(type="people_count", value=1, data={"count": 1}))["Descripción"] == "1 persona en cuadro")
check("cero", ef.fila_clara(ev(type="people_count", value=0, data={"count": 0}))["Descripción"] == "Sin personas en cuadro")
check("otras columnas vacías", solo(f, "Cantidad de personas"))

print("5. Cruce de línea")
f = ef.fila_clara(ev(type="line_cross", label="up", data={"dir": "up", "track_id": 3}))
check("hacia arriba", f["Descripción"] == "Cruce de línea hacia arriba" and f["Dirección del cruce"] == "Hacia arriba")
f = ef.fila_clara(ev(type="line_cross", label="down", data={}))
check("hacia abajo (desde la etiqueta)", f["Dirección del cruce"] == "Hacia abajo")
check("tipo traducido", f["Tipo de evento"] == "Cruce de línea")
check("alias 'cross'", ef.fila_clara(ev(type="cross", label="up"))["Tipo de evento"] == "Cruce de línea")

print("6. Gesto")
f = ef.fila_clara(ev(type="gesture", label="manos_arriba", image_path="sessions/cam0_manos.jpg"))
check("descripción", f["Descripción"] == "Gesto detectado: manos arriba", f["Descripción"])
check("gesto legible", f["Gesto"] == "Manos arriba")
check("evidencia", f["Evidencia (imagen)"] == "sessions/cam0_manos.jpg")

print("7. Objeto")
f = ef.fila_clara(ev(type="object", label="persona", value=0.87, data={"class": "person", "count": 1}))
check("descripción del ejemplo pedido", f["Descripción"] == "Objeto detectado: persona (87% de confianza)", f["Descripción"])
f = ef.fila_clara(ev(type="object", label="celular", value=0.884, data={"class": "cell phone", "count": 2,
                                                                       "bbox": [1, 2, 3, 4], "hailo_ms": 11.2}))
check("varios objetos", f["Descripción"] == "2 objetos detectados: celular (88% de confianza)", f["Descripción"])
check("columnas de objeto", f["Objeto"] == "celular" and f["Cantidad de objetos"] == 2 and f["Confianza (%)"] == 88)
check("otras columnas vacías", solo(f, "Objeto", "Cantidad de objetos", "Confianza (%)"))

print("8. Eventos heredados y datos incompletos")
f = ef.fila_clara(ev(type="gesture", label="manos_arriba", source="legacy",
                     data={"bounding_box": "{}", "person_id": None, "legacy_id": 4}))
check("origen del sistema anterior", f["Origen"] == "Sistema anterior (migrado)")
f = ef.fila_clara(ev(type="motion", value=None, data={}))
check("movimiento sin porcentaje", f["Descripción"] == "Movimiento detectado" and f["Movimiento (% del cuadro)"] == "")
f = ef.fila_clara(ev(type="object", label=None, value=None, data={"class": "dog"}))
check("objeto sin etiqueta ni confianza", f["Descripción"] == "Objeto detectado: dog" and f["Confianza (%)"] == "")
f = ef.fila_clara(ev(camera_id=None, ts=None, data=None))
check("sin cámara ni fecha no truena", f["Cámara"] == "" and f["Fecha"] == "")

print(f"\nResultado: {OK} OK, {FAIL} fallas")
sys.exit(1 if FAIL else 0)
