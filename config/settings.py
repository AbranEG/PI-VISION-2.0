# config/settings.py
# Los secretos viven en .env (no se sube a GitHub).
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

SECRET_KEY = os.getenv('SECRET_KEY', 'cambia-esta-clave')
SQLALCHEMY_DATABASE_URI = 'sqlite:///../database/pivision.db'
DEBUG = True

# IDs de tus cámaras (pueden ser índices 0, 1, 2… o cadenas con device paths)
CAMERA_IDS = [0]  # webcam USB HD 720P (/dev/video0)

EMAIL_SERVER = os.getenv('EMAIL_SERVER', 'smtp.gmail.com')
EMAIL_PORT = int(os.getenv('EMAIL_PORT', 465))
EMAIL_FROM = os.getenv('EMAIL_FROM', '')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
EMAIL_TO = os.getenv('EMAIL_TO', '')
