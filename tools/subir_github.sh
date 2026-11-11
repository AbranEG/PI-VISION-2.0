#!/usr/bin/env bash
# Respaldo local + subida de PiVision IA 2.0 a GitHub.
# Uso:  cd ~/PiVision-IA && bash tools/subir_github.sh
set -e
REPO_URL="https://github.com/AbranEG/PI-VISION-2.0.git"
cd "$(dirname "$0")/.."
echo "== Carpeta del proyecto: $(pwd)"

# 1. Respaldo completo (incluye base de datos, fotos y .env, que git no sube)
mkdir -p ~/respaldos
RESP=~/respaldos/pivision_$(date +%Y%m%d_%H%M).tar.gz
echo "== 1/5 Respaldo local en $RESP"
tar czf "$RESP" --exclude=venv --exclude=.venv --exclude=__pycache__ -C "$(dirname "$(pwd)")" "$(basename "$(pwd)")"
ls -lh "$RESP"

# 2. Identidad de git (solo si falta)
if [ -z "$(git config user.name)" ]; then read -rp "Tu nombre para git: " N; git config --global user.name "$N"; fi
if [ -z "$(git config user.email)" ]; then read -rp "Tu correo para git: " E; git config --global user.email "$E"; fi
echo "== 2/5 Autor: $(git config user.name) <$(git config user.email)>"

# 3. .gitignore: no subir datos privados; sí subir la evidencia del informe
if ! grep -q "PiVision 2.0" .gitignore 2>/dev/null; then
cat >> .gitignore <<'EOF'

# --- PiVision 2.0 ---
.venv/
sessions/
pivision.db
database/*.db
database/*.db-*
ngrok*.zip
!docs/
!docs/**
EOF
fi
echo "== 3/5 .gitignore actualizado"

# 4. Commit con revisión de seguridad
git add -A
if git diff --cached --name-only | grep -E '(^|/)\.env$|\.db$|^sessions/'; then
  echo "!! ALTO: los archivos de arriba son privados y no deben subirse. Avisa a Claude."; exit 1
fi
if git diff --cached --quiet; then echo "== 4/5 No hay cambios nuevos que guardar"
else git commit -q -m "PiVision IA 2.0: eventos, reportes, Hailo-8 (objetos y pose), optimizacion y rediseno"
     echo "== 4/5 Cambios guardados: $(git log -1 --format='%h %s')"; fi

# 5. Subir
git remote get-url nuevo >/dev/null 2>&1 || git remote add nuevo "$REPO_URL"
echo "== 5/5 Subiendo a $REPO_URL"
echo "   Usuario: tu usuario de GitHub. Contraseña: tu TOKEN (no tu contraseña normal)."
git push nuevo HEAD:main
git tag -l v1.0-baseline | grep -q . && git push nuevo v1.0-baseline || true
echo "== Listo: https://github.com/AbranEG/PI-VISION-2.0"
