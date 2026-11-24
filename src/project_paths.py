"""
Project path resolution
=======================
Makes the app and CLI scripts independent of the current working directory.

- Config files are looked up relative to the current directory first (so an
  explicit `--config my/other.yaml` behaves as users expect), then relative
  to the project root (so the default `configs/hatha_yoga_pradipika.yaml`
  works from anywhere, e.g. when launched from `src/` or an IDE).
- Relative data paths inside a config (`pdf_path`, `output_path`,
  `vector_db.persist_directory`) are resolved against the project root.
  Absolute paths are left untouched.
"""

import os
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hatha_yoga_pradipika.yaml"


def resolve_config_path(config_path: str) -> str:
    """Returns an absolute path to a config file (cwd first, then project root)."""
    path = Path(config_path).expanduser()
    if path.is_absolute() or path.exists():
        return str(path.resolve())
    return str((PROJECT_ROOT / path).resolve())


def resolve_data_path(path: str) -> str:
    """Resolves a data path from a config against the project root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())


def resolve_config_data_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a copy of `config` with its relative data paths made absolute."""
    resolved = dict(config)
    for key in ("pdf_path", "output_path"):
        if isinstance(resolved.get(key), str):
            resolved[key] = resolve_data_path(resolved[key])

    vector_db = resolved.get("vector_db")
    if isinstance(vector_db, dict) and isinstance(vector_db.get("persist_directory"), str):
        resolved["vector_db"] = dict(vector_db)
        resolved["vector_db"]["persist_directory"] = resolve_data_path(vector_db["persist_directory"])
    return resolved
