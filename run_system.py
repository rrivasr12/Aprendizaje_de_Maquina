import time
import webbrowser
import uvicorn
from threading import Thread


def open_browser():
    time.sleep(1.5)
    print("[OK] Abriendo el Frontend en el navegador web (http://127.0.0.1:8000)...")
    webbrowser.open("http://127.0.0.1:8000")


if __name__ == "__main__":
    print("=========================================================================")
    print("=== INICIANDO SISTEMA AIRPRICE ML (BACKEND FASTAPI + FRONTEND WEB UI) ===")
    print("=========================================================================")

    # Launch thread to open browser once server is ready
    Thread(target=open_browser, daemon=True).start()

    # Start FastAPI server on port 8000
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
