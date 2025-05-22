import subprocess
import os
import tempfile
import pandas as pd
from tqdm import tqdm


def predict_secondary_structure_batch(aa_sequences, output_dir=None, device="gpu"):
    # prepare output dir
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # write all sequences into one FASTA
    fasta_content = "".join(f">seq_{i}\n{seq}\n"
                             for i, seq in enumerate(aa_sequences))
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as tmp:
        fasta_input = tmp.name
        tmp.write(fasta_content.encode())

    try:
        cmd = [
            "/home/jason/outputdir/python_projects/.venv/bin/python",
            "run_model.py",
            "--outfmt", "ss2",
            "--device", device,
            fasta_input
        ]
        # launch and stream
        proc = subprocess.Popen(
            cmd,
            cwd="/home/jason/outputdir/python_projects/s4pred",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # line‑buffered
        )

        secondary_structures = []
        current_structure = []

        # one bar tick per sequence
        bar = tqdm(total=len(aa_sequences), desc="Predicting SS", unit="seq")

        for line in proc.stdout:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                # blank or separator → end of one sequence
                if current_structure:
                    secondary_structures.append("".join(current_structure))
                    current_structure = []
                    bar.update(1)
                continue
            current_structure.append(parts[2])

        # catch last sequence if no trailing blank line
        if current_structure:
            secondary_structures.append("".join(current_structure))
            bar.update(1)

        bar.close()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"run_model.py failed (exit {proc.returncode})")

    finally:
        os.remove(fasta_input)
        if output_dir.startswith(tempfile.gettempdir()):
            os.rmdir(output_dir)

    return secondary_structures

if __name__ == "__main__":
    # Example usage with a list of sequences
    aa_sequences = [
        "MVDDLYRTARPMYHEAELIREAGCPTICGVTLAN",
        "MGAAKKLREAFAR",
        "MQRPLLLSSLCPT"
    ]
    predicted_structures = predict_secondary_structure_batch(aa_sequences)
    print(predicted_structures)