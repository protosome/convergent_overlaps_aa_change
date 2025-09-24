import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Tuple
from tqdm import tqdm
from paths import ROOT_DIR


s4pred_path = ROOT_DIR / "s4pred"

S4PRED_PY   = sys.executable
S4PRED_CWD  = s4pred_path
S4PRED_EXEC = s4pred_path / "run_model_new.py"

def _write_fasta_indexed(indexed_seqs: List[Tuple[int, str]]) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", prefix="s4pred_")
    with open(tmp.name, "w") as f:
        for idx, seq in indexed_seqs:
            f.write(f">seq_{idx}\n{seq.replace(' ','').replace('\\n','')}\n")
    return tmp.name

def _parse_fas_streaming_line(line: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse one line from fas output stream.
    We expect sequence blocks in order:
      >seq_N
      AA_SEQUENCE
      SS_SEQUENCE
    This helper returns a tuple when a header appears: ("HEADER", header_text, idx)
    But we parse in the main loop instead of here, so keep simple.
    """
    return None  # placeholder — parsing handled in main loop

def _run_and_stream(
    indexed_block: List[Tuple[int, str]],
    device_flag: str,
    batch_size: int,
    amp: bool,
    pbar: tqdm,
    results: dict,
    captured_lines_limit: int = 2000
) -> Tuple[bool, str]:
    """
    Spawn subprocess for the given indexed_block (list of (orig_idx, seq_str)).
    Stream stdout line-by-line, parse fas blocks, fill `results` dict as results arrive,
    and update `pbar` for each parsed sequence.
    Returns (success:bool, captured_text_for_error_debug).
    """
    fasta_path = _write_fasta_indexed(indexed_block)
    cmd = [
        S4PRED_PY, "-u", S4PRED_EXEC,
        "--outfmt", "fas",
        "--device", device_flag,
        "--batch-size", str(batch_size),
        fasta_path
    ]
    if amp:
        # insert --amp right after outfmt/device etc (position doesn't matter much)
        cmd.insert(5, "--amp")

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TORCH_ALLOW_TF32", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

    proc = subprocess.Popen(
        cmd,
        cwd=S4PRED_CWD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )

    # state machine for parsing >seq_i  AA_LINE  SS_LINE
    expecting = None  # None | 'aa' | 'ss'
    cur_idx = None
    cur_aa = None
    captured_lines = []  # small buffer for debugging (kept limited)
    success = True

    try:
        # iterate line-by-line as they arrive
        for raw in proc.stdout:
            if raw is None:
                continue
            line = raw.rstrip("\n")
            # keep a short buffer of lines for debugging on failure
            if len(captured_lines) < captured_lines_limit:
                captured_lines.append(line)

            if not line:
                # skip blank lines
                continue
            if line.startswith("#"):
                # skip comments
                continue

            if line.startswith(">"):
                # header: expect a new AA and SS lines after this
                # extract index from header like "seq_123" (we wrote it that way)
                header = line[1:].strip()
                try:
                    if header.startswith("seq_"):
                        cur_idx = int(header.split("_", 1)[1])
                    else:
                        # fallback: try to parse trailing number
                        cur_idx = int("".join(ch for ch in header if ch.isdigit()))
                except Exception:
                    cur_idx = None
                expecting = 'aa'
                cur_aa = None
                continue

            # if we are expecting AA line
            if expecting == 'aa':
                # treat this line as AA sequence
                cur_aa = line.strip()
                expecting = 'ss'
                continue

            # if we are expecting SS line
            if expecting == 'ss':
                ss_line = line.strip()
                if cur_idx is not None:
                    # store result (ss_line should be same length as AA but we don't check here)
                    if cur_idx not in results:
                        results[cur_idx] = ss_line
                        pbar.update(1)
                    else:
                        # duplicate? ignore
                        pass
                expecting = None
                cur_idx = None
                cur_aa = None
                continue

            # ignore any other lines
            continue

        proc.wait()
        rc = proc.returncode

    finally:
        # ensure temp file removed
        try:
            os.remove(fasta_path)
        except Exception:
            pass

    captured_text = "\n".join(captured_lines)
    if rc != 0:
        # detect OOM
        lowered = captured_text.lower()
        oom = ("out of memory" in lowered) or ("cuda error: out of memory" in lowered)
        if oom:
            return False, captured_text
        # other error -> raise with captured output
        raise RuntimeError(f"run_model failed (rc={rc}). Recent output:\n{captured_text[:4000]}")
    return True, captured_text


def predict_secondary_structure_with_progress(
    aa_sequences: List[str],
    device_flag: str = "gpu",
    initial_batch_size: int = 256,
    min_batch_size: int = 1,
    amp: bool = True
) -> List[str]:
    """
    High-throughput runner with streaming tqdm progress.
    - sorts sequences by length (to reduce padding)
    - streams results from run_model_new.py updating tqdm as sequences finish
    - automatically backoffs on OOM (splits remaining work)
    """

    N = len(aa_sequences)
    # map original indices -> sequences, and create indexed list
    indexed_all = [(i, aa_sequences[i]) for i in range(N)]
    # sort by length ascending to reduce padding per minibatch inside child
    indexed_sorted = sorted(indexed_all, key=lambda t: len(t[1]))

    # results dict: orig_idx -> SS string
    results = {}

    pbar = tqdm(total=N, desc="Predicting SS", unit="seq")

    # recursive runner that handles OOM by splitting remaining work
    def _run_block(indexed_block: List[tuple], batch_size: int):
        if not indexed_block:
            return
        # Before launching, check which of these indices are already done
        remaining = [(idx, seq) for idx, seq in indexed_block if idx not in results]
        if not remaining:
            return

        success, captured = _run_and_stream(remaining, device_flag, batch_size, amp, pbar, results)
        if success:
            return
        # OOM: need to split remaining into smaller chunks and retry
        if len(remaining) <= max(1, min_batch_size):
            raise RuntimeError(f"OOM even at min_batch_size={min_batch_size}. Last captured output:\n{captured[:4000]}")
        mid = len(remaining) // 2
        left = remaining[:mid]
        right = remaining[mid:]
        # reduce batch size for subcalls (heuristic)
        new_bs = max(min_batch_size, batch_size // 2)
        _run_block(left, new_bs)
        _run_block(right, new_bs)

    # run the whole sorted list initially
    try:
        _run_block(indexed_sorted, initial_batch_size)
    finally:
        pbar.close()

    # ensure we have results for all indices
    missing = [i for i in range(N) if i not in results]
    if missing:
        raise RuntimeError(f"Missing results for indices: {missing[:20]} (total {len(missing)})")

    # assemble results in original order
    ordered = [results[i] for i in range(N)]
    return ordered


__all__ = ["predict_secondary_structure_with_progress"]
