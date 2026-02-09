#!/usr/bin/env python3
"""CLI wrapper for the notebook's 'Run Overlap' form workflow."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

import pandas as pd
from paths import ROOT_DIR


S4PRED_WEIGHTS_URL = "http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run convergent overlap generation from CLI using notebook code."
    )
    parser.add_argument(
        "--working-dir",
        default="test_results",
        help="Directory where run outputs and archives are written.",
    )
    parser.add_argument(
        "--excel-path",
        default="test_results/aa_1_aa_2.xlsx",
        help="Path to input .xlsx/.xls/.csv with aa_seq_1 and aa_seq_2 columns.",
    )
    parser.add_argument(
        "--overlap-length-selected",
        default="310",
        help="Single overlap length or comma-separated list (e.g. 310 or 310,311).",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="1-based start row in input table.",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=1,
        help="1-based end row in input table; 0 means process to end.",
    )
    parser.add_argument(
        "--use-row-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use per-row ss/sub/aln/esm weights when available.",
    )
    parser.add_argument("--ss-w", type=float, default=0.15, help="Override SS weight.")
    parser.add_argument("--sub-w", type=float, default=0.15, help="Override substitution weight.")
    parser.add_argument("--aln-w", type=float, default=0.10, help="Override alignment weight.")
    parser.add_argument("--esm-w", type=float, default=0.60, help="Override ESM weight.")
    parser.add_argument(
        "--normalize-override-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize override weights to sum to 1.",
    )
    parser.add_argument(
        "--first-pass-iters",
        type=int,
        default=1,
        help="First-pass optimization iterations.",
    )
    parser.add_argument(
        "--second-pass-iters",
        type=int,
        default=75,
        help="Second-pass optimization iterations.",
    )
    parser.add_argument(
        "--reset-caches-per-row",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, clear internal caches per row before optimization.",
    )
    parser.add_argument(
        "--archive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a zip archive for this run.",
    )
    parser.add_argument(
        "--notebook-path",
        default="convergent_overlapping_gene_generation.ipynb",
        help="Notebook path to load function definitions from.",
    )
    parser.add_argument(
        "--check-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only validate dependencies and inputs, then exit.",
    )
    parser.add_argument(
        "--setup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Install dependencies and download/extract S4PRED weights before checks/running.",
    )
    parser.add_argument(
        "--setup-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run setup steps and exit without running optimization.",
    )
    parser.add_argument(
        "--weights-url",
        default=S4PRED_WEIGHTS_URL,
        help="URL for the S4PRED weights tarball.",
    )
    parser.add_argument(
        "--force-weights-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force re-download and extract S4PRED weights even if present.",
    )
    parser.add_argument(
        "--debug-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print detailed model-loading and inference checkpoints.",
    )
    parser.add_argument(
        "--auto-setup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically run setup when required deps/weights are missing (default: enabled).",
    )
    return parser.parse_args()


def _parse_overlap_lengths(overlap_length_selected: str) -> list[int]:
    try:
        model_numbers = [int(x.strip()) for x in overlap_length_selected.split(",") if x.strip()]
        if not model_numbers:
            model_numbers = [311]
    except Exception:
        model_numbers = [311]
    return model_numbers


def _sanitize_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["aa_seq_1", "aa_seq_2", "aa_seq_1_brackets", "aa_seq_2_brackets"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\s+", "", regex=True)
        else:
            df[col] = ""
    return df


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights


def _load_notebook_runtime(notebook_path: Path) -> dict:
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    if len(code_cells) < 4:
        raise RuntimeError("Unexpected notebook layout: missing code cells.")

    deps_cell = "".join(code_cells[1].get("source", []))
    main_cell = "".join(code_cells[2].get("source", []))

    # Guard this notebook call so CLI can run even when CUDA is unavailable.
    main_cell = main_cell.replace(
        "torch.cuda.get_device_name(0)",
        (
            "torch.cuda.get_device_name(0) if torch.cuda.is_available() "
            "else print('CUDA not available; using CPU')"
        ),
    )

    namespace: dict = {"__name__": "__main__"}
    exec(compile(deps_cell, str(notebook_path), "exec"), namespace)
    exec(compile(main_cell, str(notebook_path), "exec"), namespace)
    _register_notebook_classes_for_torch_unpickle(namespace)
    return namespace


def _register_notebook_classes_for_torch_unpickle(namespace: dict) -> None:
    """
    Notebook-trained .pth files may pickle model objects as __main__.TransformerModel.
    Register these classes on the current __main__ module so torch.load can resolve them.
    """
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return

    for name in ("TransformerModel", "SinusoidalPositionalEncoding", "TransformerBlock"):
        cls = namespace.get(name)
        if cls is not None:
            setattr(main_mod, name, cls)


def _check_dependencies() -> tuple[list[str], list[str]]:
    required = {
        "torch": "torch",
        "tensorflow": "tensorflow",
        "numpy": "numpy",
        "pandas": "pandas",
        "Bio": "biopython",
        "openpyxl": "openpyxl",
        "skimage": "scikit-image",
    }
    optional = {
        "esm": "fair-esm",
    }

    missing_required: list[str] = []
    missing_optional: list[str] = []

    for module_name, pip_name in required.items():
        if importlib.util.find_spec(module_name) is None:
            missing_required.append(pip_name)
    for module_name, pip_name in optional.items():
        if importlib.util.find_spec(module_name) is None:
            missing_optional.append(pip_name)

    # De-duplicate while preserving order
    missing_required = list(dict.fromkeys(missing_required))
    missing_optional = list(dict.fromkeys(missing_optional))
    return missing_required, missing_optional


def _expected_weight_files(root_dir: Path) -> list[Path]:
    weights_dir = root_dir / "s4pred" / "weights"
    return [weights_dir / f"weights_{i}.pt" for i in range(1, 6)]


def _missing_weight_files(root_dir: Path) -> list[Path]:
    return [p for p in _expected_weight_files(root_dir) if not p.exists()]


def _download_and_extract_weights(root_dir: Path, weights_url: str) -> None:
    s4pred_dir = root_dir / "s4pred"
    if not s4pred_dir.exists():
        raise FileNotFoundError(f"s4pred directory not found at: {s4pred_dir}")

    tar_path = s4pred_dir / "weights.tar.gz"
    print(f"[SETUP] Downloading S4PRED weights from: {weights_url}")
    urllib.request.urlretrieve(weights_url, tar_path)

    print(f"[SETUP] Extracting weights archive: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=s4pred_dir)

    try:
        tar_path.unlink()
    except OSError:
        pass


def _run_setup(
    root_dir: Path,
    missing_required: list[str],
    missing_optional: list[str],
    *,
    weights_url: str,
    force_weights_download: bool,
) -> None:
    print(f"[SETUP] Root dir is: {root_dir}")

    # Mirror notebook setup packages and add anything currently missing.
    base_pkgs = ["biopython", "fair-esm"]
    install_pkgs = list(dict.fromkeys(base_pkgs + missing_required + missing_optional))
    if install_pkgs:
        print("[SETUP] Installing Python packages:", ", ".join(install_pkgs))
        subprocess.run([os.sys.executable, "-m", "pip", "install", *install_pkgs], check=True)

    missing_weights = _missing_weight_files(root_dir)
    if force_weights_download or missing_weights:
        _download_and_extract_weights(root_dir, weights_url)
    else:
        print("[SETUP] S4PRED weights already present; skipping download.")

    missing_after = _missing_weight_files(root_dir)
    if missing_after:
        raise RuntimeError(
            "S4PRED weights are still missing after setup: "
            + ", ".join(str(p) for p in missing_after)
        )
    print("[SETUP] Setup complete.")


def _remap_model_paths(runtime: dict, root_dir: Path) -> None:
    """
    Remap Colab-style model paths in runtime['model_data'] to local aa_change_models.
    """
    if "model_data" not in runtime:
        return
    model_data = runtime["model_data"]
    if "model_pth_location" not in model_data.columns:
        return

    local_models_dir = root_dir / "aa_change_models"
    remapped = 0
    missing_after = []

    new_paths = []
    for p in model_data["model_pth_location"].astype(str):
        candidate = Path(p)
        if candidate.exists():
            new_paths.append(str(candidate))
            continue

        # Colab absolute path -> local ROOT_DIR + relative tail.
        colab_prefix = "/content/convergent_overlaps_aa_change/"
        if p.startswith(colab_prefix):
            rel = p[len(colab_prefix):]
            candidate = root_dir / rel
        else:
            # Fallback: keep filename and resolve under local aa_change_models.
            candidate = local_models_dir / Path(p).name

        if str(candidate) != p:
            remapped += 1
        if not candidate.exists():
            missing_after.append(str(candidate))
        new_paths.append(str(candidate))

    model_data = model_data.copy()
    model_data["model_pth_location"] = new_paths
    runtime["model_data"] = model_data

    print(f"[INFO] Model path remap count: {remapped}")
    if missing_after:
        preview = ", ".join(missing_after[:3])
        raise RuntimeError(
            "Some model files are missing after path remap. Example(s): "
            + preview
            + f"\nExpected under: {local_models_dir}"
        )


def _validate_s4pred_runtime(runtime: dict, root_dir: Path) -> None:
    s4pred_dir = root_dir / "s4pred"
    if not s4pred_dir.exists():
        raise RuntimeError(f"s4pred directory is missing: {s4pred_dir}")

    fn = runtime.get("predict_secondary_structure")
    if fn is None:
        return
    consts = {c for c in getattr(fn, "__code__", object()).co_consts or () if isinstance(c, str)}
    legacy = "/content/convergent_overlaps_aa_change/s4pred"
    if legacy in consts:
        raise RuntimeError(
            "Legacy Colab S4PRED path detected in running_s4pred.py. "
            "Expected local ROOT_DIR/s4pred. Update running_s4pred.py to use paths.ROOT_DIR."
        )


def _print_model_checkpoint(runtime: dict, length: int) -> None:
    model_data = runtime.get("model_data")
    if model_data is None:
        print("[CHKPT] model_data not available in runtime.", file=sys.__stderr__)
        return
    if "overlap_length" not in model_data.columns or "model_pth_location" not in model_data.columns:
        print("[CHKPT] model_data missing expected columns.", file=sys.__stderr__)
        return

    rows = model_data.loc[model_data["overlap_length"] == int(length), ["overlap_length", "model_pth_location"]]
    if rows.empty:
        print(f"[CHKPT] No model_data row for overlap_length={length}.", file=sys.__stderr__)
        return
    path_str = str(rows.iloc[0]["model_pth_location"])
    p = Path(path_str)
    exists = p.exists()
    size = p.stat().st_size if exists else -1
    print(
        f"[CHKPT] overlap_length={length} model_path={path_str} exists={exists} size={size}",
        file=sys.__stderr__,
    )


def _enable_debug_checkpoints(runtime: dict) -> None:
    torch_mod = runtime.get("torch")
    if torch_mod is None:
        print("[CHKPT] torch module missing in runtime; cannot hook torch.load", file=sys.__stderr__)
        return

    orig_torch_load = torch_mod.load

    def _debug_torch_load(*args, **kwargs):
        model_path = args[0] if args else kwargs.get("f", "<unknown>")
        p = Path(str(model_path))
        exists = p.exists()
        size = p.stat().st_size if exists else -1
        print(
            f"[CHKPT] torch.load path={p} exists={exists} size={size} kwargs={list(kwargs.keys())}",
            file=sys.__stderr__,
        )
        return orig_torch_load(*args, **kwargs)

    torch_mod.load = _debug_torch_load

    if "run_inference_for_models" in runtime:
        orig_run = runtime["run_inference_for_models"]

        def _debug_run_inference_for_models(*args, **kwargs):
            model_nums = args[0] if args else kwargs.get("model_numbers")
            print(
                f"[CHKPT] run_inference_for_models start model_numbers={model_nums}",
                file=sys.__stderr__,
            )
            try:
                out = orig_run(*args, **kwargs)
                shape = getattr(out, "shape", None)
                print(f"[CHKPT] run_inference_for_models success shape={shape}", file=sys.__stderr__)
                return out
            except Exception as exc:
                print(f"[CHKPT] run_inference_for_models EXCEPTION: {exc}", file=sys.__stderr__)
                traceback.print_exc(file=sys.__stderr__)
                raise

        runtime["run_inference_for_models"] = _debug_run_inference_for_models

    print("[CHKPT] Debug checkpoints enabled.", file=sys.__stderr__)


def main() -> int:
    args = parse_args()

    root_dir = ROOT_DIR.resolve()
    notebook_path = (root_dir / args.notebook_path).resolve()
    working_dir = (root_dir / args.working_dir).resolve()
    excel_path = Path(args.excel_path)
    if not excel_path.is_absolute():
        excel_path = (root_dir / excel_path).resolve()

    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if not excel_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {excel_path}")

    missing_required, missing_optional = _check_dependencies()
    missing_weights = _missing_weight_files(root_dir)

    needs_setup = bool(missing_required or missing_weights)
    if args.auto_setup and needs_setup and not args.setup_only and not args.setup:
        print("[INFO] Missing requirements detected; running automatic setup.")
        _run_setup(
            root_dir,
            missing_required,
            missing_optional,
            weights_url=args.weights_url,
            force_weights_download=bool(args.force_weights_download),
        )
        missing_required, missing_optional = _check_dependencies()
        missing_weights = _missing_weight_files(root_dir)

    if args.setup or args.setup_only:
        _run_setup(
            root_dir,
            missing_required,
            missing_optional,
            weights_url=args.weights_url,
            force_weights_download=bool(args.force_weights_download),
        )
        missing_required, missing_optional = _check_dependencies()
        missing_weights = _missing_weight_files(root_dir)

    if args.setup_only:
        return 0

    if missing_required:
        pkg_line = " ".join(missing_required)
        raise RuntimeError(
            "Missing required dependencies for notebook runtime: "
            + ", ".join(missing_required)
            + f"\nInstall with:\n  pip install {pkg_line}"
        )
    if missing_optional:
        print(
            "[WARN] Optional packages not found (needed only for ESM/contact-map paths): "
            + ", ".join(missing_optional)
        )
    if missing_weights:
        weights_dir = root_dir / "s4pred" / "weights"
        raise RuntimeError(
            "Missing S4PRED model weights in: "
            + str(weights_dir)
            + "\nRun setup with:\n  python run_overlap_cli.py --setup --check-only"
        )

    if args.check_only:
        print("[OK] Dependency and input checks passed.")
        return 0

    model_numbers = _parse_overlap_lengths(args.overlap_length_selected)
    print(f"[INFO] model_numbers -> {model_numbers}")
    print(f"[INFO] Loading notebook runtime from: {notebook_path}")

    os.makedirs(working_dir, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(root_dir)
        runtime = _load_notebook_runtime(notebook_path)
    finally:
        os.chdir(old_cwd)

    if "load_pairs_table" not in runtime or "optimize_pair_and_save" not in runtime:
        raise RuntimeError("Missing required notebook functions: load_pairs_table or optimize_pair_and_save.")

    _remap_model_paths(runtime, root_dir)
    _validate_s4pred_runtime(runtime, root_dir)
    if args.debug_checkpoints:
        _enable_debug_checkpoints(runtime)

    runtime["RESET_CACHES_PER_ROW"] = bool(args.reset_caches_per_row)
    print(f"[INFO] RESET_CACHES_PER_ROW -> {runtime['RESET_CACHES_PER_ROW']}")

    print("[INFO] Loading pairs table...")
    pairs_df = runtime["load_pairs_table"](str(excel_path))
    pairs_df = _sanitize_pairs_df(pairs_df)

    s_row = max(1, int(args.start_row))
    e_row = int(args.end_row) if args.end_row > 0 else None
    start_idx = s_row - 1
    df_slice = pairs_df.iloc[start_idx:e_row] if e_row is not None else pairs_df.iloc[start_idx:]
    print(f"[INFO] Will process {len(df_slice)} rows (rows {s_row} to {'end' if e_row is None else e_row}).")

    override_weights = {
        "ss": float(args.ss_w),
        "sub": float(args.sub_w),
        "aln": float(args.aln_w),
        "esm": float(args.esm_w),
    }
    if args.normalize_override_weights:
        override_weights = _normalize_weights(override_weights)
    print("[INFO] Override weights (normalized if requested):", override_weights)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    overall_base = working_dir / f"convergent_overlap_run_{timestamp}"
    os.makedirs(overall_base, exist_ok=True)

    all_out_files: list[str] = []

    for length in model_numbers:
        print("\n" + "#" * 80)
        print(f"[INFO] Starting processing for overlap_length = {length}")
        length_dir = overall_base / f"len_{length}"
        os.makedirs(length_dir, exist_ok=True)

        runtime["CURRENT_OVERLAP_LENGTH"] = int(length)
        runtime["model_numbers"] = [int(length)]
        runtime["CURRENT_OUTPUT_DIR"] = str(length_dir)
        runtime["working_directory"] = str(length_dir)
        runtime["working_dir"] = str(length_dir)
        print(f"[INFO] Active model_numbers for this length: {runtime['model_numbers']}")
        if args.debug_checkpoints:
            _print_model_checkpoint(runtime, int(length))

        for i, row in df_slice.iterrows():
            row_display_num = i + 1
            print("\n" + "-" * 60)
            print(f"[ROW] length={length} | Processing row {row_display_num} (index {i}) ...")

            seq1 = row.get("aa_seq_1")
            seq2 = row.get("aa_seq_2")
            seq1_bracket = row.get("aa_seq_1_brackets", "")
            seq2_bracket = row.get("aa_seq_2_brackets", "")

            if not seq1_bracket or str(seq1_bracket).strip() == "" or str(seq1_bracket).lower() in ("nan", "none"):
                seq1_bracket = seq1
            if not seq2_bracket or str(seq2_bracket).strip() == "" or str(seq2_bracket).lower() in ("nan", "none"):
                seq2_bracket = seq2

            if not isinstance(seq1, str) or not isinstance(seq2, str) or not seq1 or not seq2:
                print(f"[WARN] Row {row_display_num}: missing/invalid sequences, skipping.")
                continue

            if args.use_row_weights:
                try:
                    row_weights = {
                        "ss": float(row.get("ss", args.ss_w)),
                        "sub": float(row.get("sub", args.sub_w)),
                        "aln": float(row.get("aln", args.aln_w)),
                        "esm": float(row.get("esm", args.esm_w)),
                    }
                    total = sum(row_weights.values())
                    if total > 0:
                        row_weights = {k: v / total for k, v in row_weights.items()}
                    joined = ", ".join([f"{k}={v:.3f}" for k, v in row_weights.items()])
                    print(f"Row {row_display_num} weights: {joined}")
                except Exception as exc:
                    print(f"[WARN] Row {row_display_num}: invalid weights, falling back to override. {exc}")
                    row_weights = override_weights
            else:
                row_weights = override_weights

            try:
                out_path = runtime["optimize_pair_and_save"](
                    seq1,
                    seq2,
                    seq1_bracket,
                    seq2_bracket,
                    row_display_num,
                    row_weights,
                    first_pass_iterations=args.first_pass_iters,
                    second_pass_iterations=args.second_pass_iters,
                )
                if out_path:
                    all_out_files.append(out_path)
                    print(f"[OK] Wrote: {out_path}")
            except Exception:
                print(f"[ERROR] Row {row_display_num}: exception during optimize_pair_and_save:")
                traceback.print_exc()

    if args.archive:
        try:
            archive_base = working_dir / f"convergent_overlap_outputs_{timestamp}"
            overall_archive = shutil.make_archive(str(archive_base), "zip", root_dir=str(overall_base))
            print("\n[DONE] Created overall archive:", overall_archive)
        except Exception:
            print("\n[WARN] Failed to create overall archive:")
            traceback.print_exc()

    print("\n[FINISHED] Run complete.")
    print(f"[INFO] Individual output files generated: {len(all_out_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
