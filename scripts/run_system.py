import os
import sys
import time
import webbrowser
from pathlib import Path
import uvicorn
from threading import Thread

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def open_browser():
    time.sleep(1.8)
    print("[OK] Servidor iniciado. Abriendo Frontend en el navegador web (http://127.0.0.1:8000)...")
    webbrowser.open("http://127.0.0.1:8000")


def start():
    print("=========================================================================")
    print("=== INICIANDO SISTEMA AIRPRICE ML (BACKEND FASTAPI + FRONTEND WEB UI) ===")
    print("=========================================================================")

    models_exist = (BASE_DIR / "models" / "best_rf_model.joblib").exists()
    if not models_exist:
        print("[AVISO] No se encontraron modelos en la carpeta 'models/'.")
        print("Ejecutando pipeline de entrenamiento 'scripts/train.py'...")
        os.system(f'"{sys.executable}" "{BASE_DIR / "scripts" / "train.py"}"')

    # Launch background thread to open browser
    Thread(target=open_browser, daemon=True).start()

    # Start FastAPI Uvicorn server on port 8000
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    start()
