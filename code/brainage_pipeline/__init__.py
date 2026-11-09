"""Participant-nested multimodal brain-age prediction pipeline."""
from pathlib import Path
import sys

FINAL = Path(__file__).resolve().parents[1] / "final"
if str(FINAL) not in sys.path:
    sys.path.insert(0, str(FINAL))
