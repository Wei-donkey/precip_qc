"""Path bootstrap so scripts/ can import the core modules in src/ as `src.<module>`."""
from pathlib import Path
import sys

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
