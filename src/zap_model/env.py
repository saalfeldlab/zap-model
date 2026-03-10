"""Environment variable defaults. Loads .env on import."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env_path(var: str) -> str:
    """Return env var value if set, else ``...`` (pydantic required sentinel)."""
    val = os.environ.get(var)
    return val if val is not None else ...  # type: ignore[return-value]


def env_derived_path(var: str, *relative: str) -> str:
    """Return root / relative if env var is set, else ``...``."""
    val = os.environ.get(var)
    return str(Path(val, *relative)) if val is not None else ...  # type: ignore[return-value]
