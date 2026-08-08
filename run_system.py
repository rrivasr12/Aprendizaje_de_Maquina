import sys
from pathlib import Path

# Add scripts directory to path and delegate to scripts.run_system
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from scripts.run_system import start

if __name__ == "__main__":
    start()
