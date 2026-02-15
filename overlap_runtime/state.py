from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RuntimeState:
    """Mutable runtime context shared by CLI and notebook wrapper."""

    root_dir: Path
    esm_device: str = "auto"
    esm_batch_size: int = 4
    esm_autocast: bool = False
    require_esm: bool = True
    reset_caches_per_row: bool = False
    debug_checkpoints: bool = False
    extras: dict[str, Any] = field(default_factory=dict)
