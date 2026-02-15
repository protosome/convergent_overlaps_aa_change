from __future__ import annotations

import pandas as pd


def sanitize_pairs_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["aa_seq_1", "aa_seq_2", "aa_seq_1_brackets", "aa_seq_2_brackets"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"\s+", "", regex=True)
        else:
            df[col] = ""
    return df


def compute_row_slice(df: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    s_row = max(1, int(start_row))
    e_row = int(end_row) if int(end_row) > 0 else None
    if e_row is not None and e_row < s_row:
        raise ValueError(
            f"Invalid row range: start_row={s_row}, end_row={e_row}. "
            "Use end_row=0 to process to the end."
        )
    start_idx = s_row - 1
    return df.iloc[start_idx:e_row] if e_row is not None else df.iloc[start_idx:]


def load_pairs_table(path: str):
    from .notebook_core import load_pairs_table as _load_pairs_table

    return _load_pairs_table(path)
