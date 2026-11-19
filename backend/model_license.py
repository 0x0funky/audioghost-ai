"""
Meta's SAM License, which comes with the SAM-Audio weights.

AudioGhost redistributes the weights (model_hub/), and the license asks that a
copy goes with them. The UI shows it once, before the first download from
AudioGhost's copies, and remembers the answer (per installation).
"""
import json
import time
from pathlib import Path

from settings import DATA_DIR

TEXT_FILE = Path(__file__).resolve().parent / "licenses" / "SAM_LICENSE.txt"
ACCEPTED_FILE = DATA_DIR / "sam_license_accepted.json"
#: The "Last Updated" date of the text in TEXT_FILE. A new version asks again.
VERSION = "2025-11-19"
URL = "https://github.com/facebookresearch/sam-audio/blob/main/LICENSE"


def text() -> str:
    return TEXT_FILE.read_text(encoding="utf-8")


def accepted() -> bool:
    try:
        return json.loads(ACCEPTED_FILE.read_text(encoding="utf-8")).get("version") == VERSION
    except (OSError, ValueError):
        return False


def accept() -> None:
    ACCEPTED_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACCEPTED_FILE.write_text(json.dumps({"version": VERSION, "accepted_at": time.time()}), encoding="utf-8")
