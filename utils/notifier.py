import os
import time
import json
from utils.email_alert import send_email_alert
from utils.telegram_alert import send_telegram_alert

CONFIG_PATH = os.path.join("config", "notifications.json")
_last_email_time = 0  # control interno de cooldown

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {
            "email_enabled": False,
            "email_to": "",
            "telegram_enabled": False,
            "telegram_chat_id": "",
            "cooldown_seconds": 60
        }
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def send_notification_if_gesture(image_path):
    global _last_email_time
    config = load_config()
    now = time.time()

    cooldown = config.get("cooldown_seconds", 60)
    if now - _last_email_time < cooldown:
        print(f"[Notificaciones] Cooldown activo ({cooldown}s). No se enviará notificación.")
        return

    if not os.path.exists(image_path):
        print(f"[Notificaciones] Imagen no encontrada: {image_path}")
        return

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except Exception as e:
        print(f"[Notificaciones] Error leyendo imagen: {e}")
        return

    subject = "🔔 Gesto detectado - PiVision IA"
    body    = "Se ha detectado un gesto (ej. manos arriba). Imagen adjunta."

    # 📨 Enviar correo
    if config.get("email_enabled", False):
        to_email = config.get("email_to", "").strip()
        if to_email:
            email_config = {
                "EMAIL_SERVER": os.getenv("EMAIL_SERVER", "smtp.gmail.com"),
                "EMAIL_PORT": int(os.getenv("EMAIL_PORT", 465)),
                "EMAIL_FROM": os.getenv("EMAIL_FROM", ""),
                "EMAIL_PASSWORD": os.getenv("EMAIL_PASSWORD", "")
            }
            send_email_alert(subject, body, image_data, to_email, email_config)
        else:
            print("[Correo] ❗ No se ha configurado un correo de destino.")

    # 📲 Enviar Telegram
    if config.get("telegram_enabled", False):
        token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = config.get("telegram_chat_id", "").strip()
        if chat_id:
            send_telegram_alert(token, chat_id, subject, image_path)
        else:
            print("[Telegram] ❗ No se ha configurado un chat_id de destino.")

    _last_email_time = now
