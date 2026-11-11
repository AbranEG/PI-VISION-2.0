# app/export_format.py
"""
Presentación de eventos para la exportación CSV "clara" (para usuario final).

Solo transforma filas ya leídas de la base de datos: no consulta, no escribe y no
toca la detección. La exportación técnica (campos originales + JSON) sigue en
/api/v1/events/export_full.csv.
"""

TIPOS = {
    "motion": "Movimiento",
    "face": "Rostro",
    "gesture": "Gesto",
    "line_cross": "Cruce de línea",
    "cross": "Cruce de línea",
    "object": "Objeto",
    "people_count": "Personas",
}

DIRECCIONES = {"up": "Hacia arriba", "down": "Hacia abajo"}

FUENTES_CONTEO = {"hailo": "detección por cuerpo", "rostros": "detección por rostro"}

COLUMNAS = [
    "Fecha", "Hora", "Cámara", "Tipo de evento", "Descripción",
    "Persona identificada", "Cantidad de personas", "Movimiento (% del cuadro)",
    "Objeto", "Cantidad de objetos", "Confianza (%)", "Dirección del cruce", "Gesto",
    "Estado", "Notas", "Evidencia (imagen)", "Origen", "ID del evento",
]


def _num(v):
    """Número o None (acepta cadenas numéricas)."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_num(v, decimals=1):
    """2.0 → '2', 2.345 → '2.3' (sin ceros sobrantes)."""
    if v is None:
        return ""
    s = f"{v:.{decimals}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _texto_gesto(label):
    return (label or "gesto").replace("_", " ").strip().capitalize()


def _fecha_hora(ts):
    ts = str(ts or "")
    if len(ts) < 10:
        return ts, ""
    fecha = f"{ts[8:10]}/{ts[5:7]}/{ts[0:4]}"
    hora = ts[11:19] if len(ts) >= 19 else ts[11:]
    return fecha, hora


def fila_clara(ev):
    """
    Convierte un evento (dict de EventStore.query) en un dict con las COLUMNAS claras.
    Los campos que no aplican al tipo de evento quedan vacíos.
    """
    data = ev.get("data") if isinstance(ev.get("data"), dict) else {}
    tipo = ev.get("type") or ""
    label = ev.get("label")
    value = _num(ev.get("value"))
    fecha, hora = _fecha_hora(ev.get("ts"))
    cam = ev.get("camera_id")

    out = {c: "" for c in COLUMNAS}
    out.update({
        "Fecha": fecha,
        "Hora": hora,
        "Cámara": f"Cámara {cam}" if cam not in (None, "") else "",
        "Tipo de evento": TIPOS.get(tipo, tipo.replace("_", " ").capitalize()),
        "Estado": "Revisado" if ev.get("reviewed") else "Pendiente",
        "Notas": ev.get("notes") or "",
        "Evidencia (imagen)": ev.get("image_path") or "",
        "Origen": "Sistema anterior (migrado)" if ev.get("source") == "legacy" else "Sistema actual",
        "ID del evento": ev.get("id", ""),
    })

    if tipo == "motion":
        pct = _num(data.get("area_pct"))
        pct = value if pct is None else pct
        out["Movimiento (% del cuadro)"] = _fmt_num(pct)
        out["Descripción"] = (f"Movimiento detectado en {_fmt_num(pct)}% del cuadro"
                              if pct is not None else "Movimiento detectado")

    elif tipo == "face":
        nombre = label or "Desconocido"
        out["Persona identificada"] = nombre
        out["Descripción"] = ("Rostro desconocido detectado" if nombre == "Desconocido"
                              else f"Rostro reconocido: {nombre}")

    elif tipo == "people_count":
        n = _num(data.get("count"))
        n = value if n is None else n
        n = int(n) if n is not None else None
        out["Cantidad de personas"] = "" if n is None else n
        fuente = FUENTES_CONTEO.get(data.get("fuente"))
        if n is None:
            desc = "Cambio en el conteo de personas"
        elif n == 0:
            desc = "Sin personas en cuadro"
        elif n == 1:
            desc = "1 persona en cuadro"
        else:
            desc = f"{n} personas en cuadro"
        out["Descripción"] = f"{desc} ({fuente})" if fuente else desc

    elif tipo in ("line_cross", "cross"):
        d = data.get("dir") or label
        direccion = DIRECCIONES.get(d, "")
        out["Dirección del cruce"] = direccion
        out["Descripción"] = f"Cruce de línea {direccion.lower()}" if direccion else "Cruce de línea"

    elif tipo == "gesture":
        g = _texto_gesto(label or data.get("gesture"))
        out["Gesto"] = g
        out["Descripción"] = f"Gesto detectado: {g.lower()}"

    elif tipo == "object":
        objeto = label or data.get("class") or "objeto"
        cant = _num(data.get("count"))
        cant = int(cant) if cant else 1
        conf = round(value * 100) if value is not None else None
        out["Objeto"] = objeto
        out["Cantidad de objetos"] = cant
        out["Confianza (%)"] = "" if conf is None else conf
        conf_txt = f" ({conf}% de confianza)" if conf is not None else ""
        out["Descripción"] = (f"Objeto detectado: {objeto}{conf_txt}" if cant == 1
                              else f"{cant} objetos detectados: {objeto}{conf_txt}")

    else:
        out["Descripción"] = label or out["Tipo de evento"]

    return out
