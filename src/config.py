"""Centralized config loader.

Example:
>>> from config import CONFIG
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
with _CONFIG_PATH.open() as f:
    CONFIG = json.load(f)
