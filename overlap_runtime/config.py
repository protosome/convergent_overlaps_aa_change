from __future__ import annotations

from dataclasses import dataclass


S4PRED_WEIGHTS_URL = (
    "https://bioinf.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz"
)
ALLOWED_PLOT_METRICS = ("Combined", "ESM_avg", "SS_avg", "Align1", "Align2", "Sub1", "Sub2")


@dataclass(frozen=True)
class ESMConfig:
    device: str = "auto"
    batch_size: int = 4
    autocast: bool = False
    require_esm: bool = True
