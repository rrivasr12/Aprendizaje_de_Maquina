import glob
import subprocess
import sys
from pathlib import Path


def run_notebooks():
    notebook_dir = Path(__file__).resolve().parent.parent / "notebooks"
    notebooks = sorted(glob.glob(str(notebook_dir / "*.ipynb")))

    if not notebooks:
        print("⚠️ No se encontraron notebooks en la carpeta notebooks/")
        return

    print(f"📂 Encontrados {len(notebooks)} notebooks en {notebook_dir}:")
    for nb in notebooks:
        print(f"  - {Path(nb).name}")

    print("\n=========================================================")
    print("=== INICIANDO EJECUCIÓN DE NOTEBOOKS (NBCONVERT)      ===")
    print("=========================================================")

    success_count = 0
    error_count = 0

    for nb in notebooks:
        nb_name = Path(nb).name
        print(f"\n🚀 Ejecutando notebook: {nb_name}...")
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutionPreprocessor.timeout=600",
            nb,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {nb_name} ejecutado y guardado correctamente.")
            success_count += 1
        else:
            print(f"❌ Error al ejecutar {nb_name}:")
            print(result.stderr)
            error_count += 1

    print("\n=========================================================")
    print(f"=== RESUMEN: {success_count} exitosos, {error_count} con errores. ===")
    print("=========================================================")


if __name__ == "__main__":
    run_notebooks()
