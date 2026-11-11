#!/usr/bin/env bash
# Arranca PiVision IA completo:
#   1) servicio Hailo (Python 3.13 del sistema) en segundo plano, log en log/hailo_service.log
#   2) servidor Flask (venv, Python 3.11) en primer plano
# Ctrl+C detiene ambos.
#
# Uso:  cd ~/PiVision-IA && bash start_pivision.sh
#       bash start_pivision.sh --sin-hailo     (solo la app, sin detección de objetos)
set -e
cd "$(dirname "$0")"
mkdir -p log
SOCK="${HAILO_SOCKET:-/tmp/pivision-hailo.sock}"
export HAILO_SOCKET="$SOCK"
HAILO_PID=""

limpiar() {
  if [ -n "$HAILO_PID" ] && kill -0 "$HAILO_PID" 2>/dev/null; then
    echo ">> Deteniendo servicio Hailo (pid $HAILO_PID)"
    kill "$HAILO_PID" 2>/dev/null || true
    wait "$HAILO_PID" 2>/dev/null || true
  fi
}
trap limpiar EXIT INT TERM

if [ "$1" != "--sin-hailo" ]; then
  if [ -S "$SOCK" ] && /usr/bin/python3 -c "
import sys; sys.path.insert(0, '.')
from hailo_service.protocol import connect, send_msg, recv_msg
s = connect('$SOCK'); send_msg(s, {'op': 'ping'}); recv_msg(s)" 2>/dev/null; then
    echo ">> El servicio Hailo ya estaba corriendo en $SOCK"
  else
    echo ">> Iniciando servicio Hailo…"
    /usr/bin/python3 -m hailo_service.server --socket "$SOCK" >> log/hailo_service.log 2>&1 &
    HAILO_PID=$!
    for i in $(seq 1 50); do [ -S "$SOCK" ] && break; sleep 0.2; done
    if [ -S "$SOCK" ]; then
      echo ">> Servicio Hailo listo (pid $HAILO_PID)"
    else
      echo "!! El servicio Hailo no arrancó; revisa log/hailo_service.log. La app sigue sin objetos."
    fi
  fi
fi

echo ">> Iniciando servidor PiVision…"
# shellcheck disable=SC1091
source venv/bin/activate
python server.py
