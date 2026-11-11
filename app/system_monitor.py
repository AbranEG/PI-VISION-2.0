# app/system_monitor.py

import psutil
import shutil
import datetime

def get_system_metrics():
    """
    Retorna las métricas básicas del sistema en forma de diccionario.
    Incluye uso de CPU, memoria RAM, y disco principal.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=None)  # no bloquea (antes 0.5 s por lectura)
        memory = psutil.virtual_memory()
        disk = shutil.disk_usage("/")

        temp = None
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                temp = round(int(f.read()) / 1000.0, 1)   # °C del procesador (Raspberry Pi)
        except Exception:
            pass

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "metrics",
            "cpu": round(cpu_percent, 1),              # en %
            "memory": round(memory.percent, 1),        # en %
            "disk": round(disk.used / disk.total * 100, 1),  # en %
            "temp": temp                               # en °C (None si no se puede leer)
        }
    except Exception as e:
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "metrics",
            "error": str(e)
        }
