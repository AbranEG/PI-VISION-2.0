#!/usr/bin/env bash
# Prepara la copia del código ORIGINAL (commit 7b85e78) para medir el "antes".
# No toca ~/PiVision-IA salvo para crear la etiqueta y el worktree.
set -e
REPO=~/PiVision-IA
BASE=~/PiVision-baseline
cd "$REPO"

echo ">> Etiqueta v1.0-baseline en 7b85e78"
git rev-parse -q --verify v1.0-baseline >/dev/null || git tag v1.0-baseline 7b85e78

echo ">> Copia del código original en $BASE"
if [ ! -e "$BASE/.git" ]; then
  # Si quedó una carpeta sin código (p. ej. solo results/ vacío), se quita
  if [ -d "$BASE" ] && [ ! -d "$BASE/app" ]; then
    find "$BASE" -type f | grep -q . && { echo "ERROR: $BASE tiene archivos y no es la copia; revísalo a mano"; exit 1; }
    rm -rf "$BASE"
  fi
  git worktree add "$BASE" v1.0-baseline
fi

echo ">> Insumos mínimos para que el original arranque (no modifican su código)"
cp -n models/mobilefacenet.tflite "$BASE/models/"
cat > "$BASE/.env" <<'EOF'
USE_TPU=false
FACE_MATCH_THRESHOLD=0.45
GESTURE_DETECTION_CONFIDENCE=0.7
DETECTION_FRAME_SKIP=2
EOF
( cd "$BASE" && "$REPO/venv/bin/python" init_db.py )

echo ">> Listo. Código original en $BASE ($(git -C "$BASE" describe --tags))"
