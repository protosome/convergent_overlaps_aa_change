from __future__ import annotations

from . import notebook_core as core
from pathlib import Path

load_model = core.load_model
run_inference_for_models = core.run_inference_for_models


def remap_model_paths(model_data, root_dir: Path):
    """Remap Colab-trained model paths to local aa_change_models paths."""
    if "model_pth_location" not in model_data.columns:
        return model_data

    local_models_dir = Path(root_dir) / "aa_change_models"
    new_paths = []
    for p in model_data["model_pth_location"].astype(str):
        candidate = Path(p)
        if candidate.exists():
            new_paths.append(str(candidate))
            continue

        colab_prefix = "/content/convergent_overlaps_aa_change/"
        if p.startswith(colab_prefix):
            rel = p[len(colab_prefix):]
            candidate = Path(root_dir) / rel
        else:
            candidate = local_models_dir / Path(p).name
        new_paths.append(str(candidate))

    out = model_data.copy()
    out["model_pth_location"] = new_paths
    return out


__all__ = ["load_model", "run_inference_for_models", "remap_model_paths"]
