# paths.py
from pathlib import Path
import argparse

def resolve_root_dir(cli_root: str | None = None) -> Path:
    """
    Figure out where the project root is.
    - If a path is passed on the command line with --root, use that.
    - If running as a script, use the script's directory.
    - If in a notebook/interactive, fall back to the current working directory.
    """
    if cli_root:
        return Path(cli_root).expanduser().resolve()
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()

# Parse --root if present (safe even inside notebooks)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--root", help="Project root directory")
args, _ = parser.parse_known_args()

ROOT_DIR: Path = resolve_root_dir(args.root)