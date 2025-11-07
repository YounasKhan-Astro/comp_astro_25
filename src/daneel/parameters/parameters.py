# src/daneel/parameters/parameters.py
import yaml
from pathlib import Path
from typing import Any, Dict

class Parameters:
    """
    Minimal, robust parameters loader.

    Usage:
        p = Parameters("params.yaml")
        p.params  # dict with your YAML keys/values
    """
    def __init__(self, path: str):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.path}")

        with self.path.open("r") as f:
            data = yaml.safe_load(f)  # may return None for empty files

        # Always set self.params first, before any access.
        self.params: Dict[str, Any] = data or {}

        # (Optional) place for validation of required keys, e.g.:
        # for required in ["per", "inc"]:
        #     if required not in self.params:
        #         raise ValueError(f"Missing required key: {required}")

