#!/usr/bin/env python3
"""Legacy CLI shim.

This entrypoint is kept for backward compatibility and delegates to the
modular runtime CLI.
"""

from __future__ import annotations

from run_overlap_cli_modular import main as modular_main


if __name__ == "__main__":
    raise SystemExit(modular_main())
