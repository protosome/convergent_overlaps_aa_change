from __future__ import annotations

from pathlib import Path
from .io_pairs import sanitize_pairs_df, compute_row_slice


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total > 0:
        return {k: v / total for k, v in weights.items()}
    return weights


def parse_overlap_lengths(overlap_length_selected: str | list[int] | tuple[int, ...] | int) -> list[int]:
    try:
        if isinstance(overlap_length_selected, str):
            values = [int(x.strip()) for x in overlap_length_selected.split(",") if x.strip()]
        elif isinstance(overlap_length_selected, (list, tuple)):
            values = [int(x) for x in overlap_length_selected]
        else:
            values = [int(overlap_length_selected)]
    except Exception:
        values = [311]
    return values or [311]


def run_batch_from_table(
    *,
    excel_path: str,
    working_dir: str,
    overlap_length_selected: str | list[int] | tuple[int, ...] | int,
    start_row: int,
    end_row: int,
    use_row_weights: bool,
    ss_w: float,
    sub_w: float,
    aln_w: float,
    esm_w: float,
    normalize_override_weights: bool,
    first_pass_iters: int,
    second_pass_iters: int,
    archive: bool = True,
) -> list[str]:
    import os
    import shutil
    import time
    from . import notebook_core as core

    pairs_df = sanitize_pairs_df(core.load_pairs_table(excel_path))
    df_slice = compute_row_slice(pairs_df, start_row, end_row)

    override_weights = {"ss": float(ss_w), "sub": float(sub_w), "aln": float(aln_w), "esm": float(esm_w)}
    if normalize_override_weights:
        override_weights = normalize_weights(override_weights)

    model_numbers = parse_overlap_lengths(overlap_length_selected)
    os.makedirs(working_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    overall_base = Path(working_dir) / f"convergent_overlap_run_{timestamp}"
    os.makedirs(overall_base, exist_ok=True)

    all_out_files: list[str] = []
    for length in model_numbers:
        length_dir = overall_base / f"len_{length}"
        os.makedirs(length_dir, exist_ok=True)
        core.CURRENT_OVERLAP_LENGTH = int(length)
        core.model_numbers = [int(length)]
        core.CURRENT_OUTPUT_DIR = str(length_dir)
        core.working_directory = str(length_dir)
        core.working_dir = str(length_dir)

        for i, row in df_slice.iterrows():
            row_display_num = i + 1
            seq1 = row.get("aa_seq_1")
            seq2 = row.get("aa_seq_2")
            seq1_bracket = row.get("aa_seq_1_brackets", "")
            seq2_bracket = row.get("aa_seq_2_brackets", "")
            if not seq1_bracket or str(seq1_bracket).strip() == "" or str(seq1_bracket).lower() in ("nan", "none"):
                seq1_bracket = seq1
            if not seq2_bracket or str(seq2_bracket).strip() == "" or str(seq2_bracket).lower() in ("nan", "none"):
                seq2_bracket = seq2
            if not isinstance(seq1, str) or not isinstance(seq2, str) or not seq1 or not seq2:
                continue

            if use_row_weights:
                try:
                    row_weights = {
                        "ss": float(row.get("ss", ss_w)),
                        "sub": float(row.get("sub", sub_w)),
                        "aln": float(row.get("aln", aln_w)),
                        "esm": float(row.get("esm", esm_w)),
                    }
                    total = sum(row_weights.values())
                    if total > 0:
                        row_weights = {k: v / total for k, v in row_weights.items()}
                except Exception:
                    row_weights = override_weights
            else:
                row_weights = override_weights

            out_path = core.optimize_pair_and_save(
                seq1,
                seq2,
                seq1_bracket,
                seq2_bracket,
                row_display_num,
                row_weights,
                first_pass_iterations=first_pass_iters,
                second_pass_iterations=second_pass_iters,
            )
            if out_path:
                all_out_files.append(out_path)

    if archive:
        archive_base = Path(working_dir) / f"convergent_overlap_outputs_{timestamp}"
        shutil.make_archive(str(archive_base), "zip", root_dir=str(overall_base))

    return all_out_files
