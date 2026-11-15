"""Make the app's modules importable in tests (the app folder is deployed on its own)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "app"))
