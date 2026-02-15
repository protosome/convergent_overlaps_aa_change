from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import MutableMapping

from .state import RuntimeState


class RuntimeNamespace(MutableMapping):
    """Dict-like adapter that proxies keys to module globals."""

    def __init__(self, module):
        self._module = module

    def __getitem__(self, key):
        return getattr(self._module, key)

    def __setitem__(self, key, value):
        setattr(self._module, key, value)

    def __delitem__(self, key):
        delattr(self._module, key)

    def __iter__(self):
        return iter(self._module.__dict__)

    def __len__(self):
        return len(self._module.__dict__)

    def __contains__(self, key):
        return hasattr(self._module, key)


def _register_notebook_classes_for_torch_unpickle(core_module) -> None:
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for name in ("TransformerModel", "SinusoidalPositionalEncoding", "TransformerBlock"):
        cls = getattr(core_module, name, None)
        if cls is not None:
            setattr(main_mod, name, cls)


def create_runtime(
    root_dir: Path,
    *,
    esm_device: str = "auto",
    esm_batch_size: int = 4,
    esm_autocast: bool = False,
    require_esm: bool = True,
    reset_caches_per_row: bool = False,
    debug_checkpoints: bool = False,
):
    from . import notebook_core as core

    state = RuntimeState(
        root_dir=Path(root_dir).resolve(),
        esm_device=esm_device,
        esm_batch_size=esm_batch_size,
        esm_autocast=esm_autocast,
        require_esm=require_esm,
        reset_caches_per_row=reset_caches_per_row,
        debug_checkpoints=debug_checkpoints,
    )

    core.ROOT_DIR = state.root_dir
    core.RESET_CACHES_PER_ROW = bool(reset_caches_per_row)
    _register_notebook_classes_for_torch_unpickle(core)

    runtime = RuntimeNamespace(core)
    runtime["_runtime_state"] = state
    return runtime


def load_pairs_table(path: str):
    from .notebook_core import load_pairs_table as _load_pairs_table

    return _load_pairs_table(path)


def optimize_pair_and_save(*args, **kwargs):
    from .notebook_core import optimize_pair_and_save as _optimize_pair_and_save

    return _optimize_pair_and_save(*args, **kwargs)


def run_batch_from_table(*args, **kwargs):
    from .pipeline import run_batch_from_table as _run_batch_from_table

    return _run_batch_from_table(*args, **kwargs)
