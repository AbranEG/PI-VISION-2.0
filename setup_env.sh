#!/usr/bin/env bash
# Crea ~/PiVision-IA/venv con Python 3.11 (vía uv, sin conda) e instala dependencias.
set -e
cd ~/PiVision-IA

echo ">> Paquetes del sistema"
sudo apt update
sudo apt install -y v4l-utils git curl

echo ">> uv"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo ">> Entorno venv con Python 3.11"
[ -x venv/bin/python ] || uv venv --python 3.11 --python-preference only-managed venv
venv/bin/python --version

echo ">> Dependencias de Python"
uv pip install --python venv/bin/python -r requirements-pi.txt

echo ">> Modelo MobileFaceNet"
MODEL=models/mobilefacenet.tflite
SHA=be4bc7cfc53f7bc336d0f28b1ab92535f618c913a422b683210750f6b5354854
[ -f "$MODEL" ] || curl -fsSL -o "$MODEL" https://raw.githubusercontent.com/MCarlomagno/FaceRecognitionAuth/master/assets/mobilefacenet.tflite
echo "$SHA  $MODEL" | sha256sum -c -

echo ">> Carpetas de trabajo"
mkdir -p database log sessions
[ -f database/pivision.db ] || cp pivision.db database/pivision.db

echo ">> .env"
if [ ! -f .env ]; then
  cp env.example .env
  sed -i "s|^SECRET_KEY=.*|SECRET_KEY=$(openssl rand -hex 32)|" .env
fi
chmod 600 .env

echo ">> Verificación"
venv/bin/python - <<'EOF'
import cv2, mediapipe, flask, numpy
from face_encoder import FaceEncoder
v = FaceEncoder().encode(numpy.zeros((112,112,3), numpy.uint8))
print("OK  opencv", cv2.__version__, "| mediapipe", mediapipe.__version__, "| numpy", numpy.__version__, "| embedding", v.shape)
EOF
echo ">> Listo. Activa con: conda deactivate; source ~/PiVision-IA/venv/bin/activate"
