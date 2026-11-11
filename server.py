# server.py
import os
from dotenv import load_dotenv
load_dotenv()
from app import create_app

app = create_app()

if __name__ == '__main__':
    # Hallazgo 13: antes corría siempre con debug=True. Con el depurador de Werkzeug activo,
    # cualquiera que llegue a la página de error (por ejemplo a través de ngrok) puede
    # ejecutar código en la Pi. Ahora está apagado salvo que se pida en .env (FLASK_DEBUG=true).
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    if debug:
        print("[AVISO] FLASK_DEBUG=true: modo depuración activo. No lo expongas a internet.")
    # Sin reloader para que solo haya un único proceso (una sola captura por cámara)
    app.run(
        debug=debug,
        host='0.0.0.0',
        port=int(os.getenv("PORT", 5000)),
        use_reloader=False,
        threaded=True
    )
