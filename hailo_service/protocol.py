# hailo_service/protocol.py
"""
Protocolo entre la app (Python 3.11) y el servicio Hailo (Python 3.13).

Cada mensaje viaja por un socket Unix local:
    [4 bytes: longitud del encabezado, big-endian] [encabezado JSON] [carga binaria opcional]
El encabezado declara "payload" = número de bytes binarios que siguen (0 si no hay).

Peticiones:
    {"op": "ping"}
    {"op": "info"}
    {"op": "detect", "shape": [alto, ancho, 3], "min_score": 0.5, "classes": ["person", ...]}
        + payload = frame BGR uint8 (alto*ancho*3 bytes)
Respuestas:
    {"ok": true, ...}  |  {"ok": false, "error": "..."}

Solo usa la biblioteca estándar y numpy, para que ambos lados lo importen.
"""
import json
import os
import socket
import struct

DEFAULT_SOCKET = os.getenv("HAILO_SOCKET", "/tmp/pivision-hailo.sock")
MAX_HEADER = 64 * 1024
MAX_PAYLOAD = 64 * 1024 * 1024


def _recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("conexión cerrada")
        got += k
    return bytes(buf)


def send_msg(sock, header, payload=b""):
    header = dict(header, payload=len(payload))
    h = json.dumps(header, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(">I", len(h)) + h)
    if payload:
        sock.sendall(payload)


def recv_msg(sock):
    (hlen,) = struct.unpack(">I", _recv_exact(sock, 4))
    if hlen > MAX_HEADER:
        raise ValueError("encabezado demasiado grande")
    header = json.loads(_recv_exact(sock, hlen).decode("utf-8"))
    n = int(header.get("payload", 0))
    if n > MAX_PAYLOAD:
        raise ValueError("carga demasiado grande")
    payload = _recv_exact(sock, n) if n else b""
    return header, payload


def connect(path=DEFAULT_SOCKET, timeout=2.0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(path)
    return s
