import subprocess
import os
import tempfile
import sys
from paths import ROOT_DIR


S4PRED_CWD = ROOT_DIR / "s4pred"


def predict_secondary_structure(aa_sequence, output_dir=None, device="gpu"):
    # If no output directory is provided, use a temporary directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Create a FASTA-format string from the amino acid sequence
    fasta_content = f">input_sequence\n{aa_sequence}\n"
    
    # Create a temporary FASTA file to store the sequence
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_fasta:
        fasta_input = temp_fasta.name
        temp_fasta.write(fasta_content.encode())
    
    try:
        # Define the command with a custom prefix for the output file
        # Need to identify location of python venv, will access dependencies there

        cmd = [
            sys.executable, "run_model.py",
            "--outfmt", "ss2",
            "--device", device,
            fasta_input
        ]

        # Capture the output using subprocess and extract only the secondary structure
        result = subprocess.run(cmd, cwd=str(S4PRED_CWD),
                                capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            raise RuntimeError(f"run_model.py failed with exit code {result.returncode}")
        
        # Extract the secondary structure from the command output
        secondary_structure = []
        for line in result.stdout.splitlines():
            if line.startswith("#") or len(line.split()) < 3:
                continue  # Skip comment lines and lines with insufficient columns
            parts = line.split()
            secondary_structure.append(parts[2])  # Column 3 contains the secondary structure

        # Concatenate the secondary structure list into a single string
        predicted_structure = "".join(secondary_structure)

    finally:
        # Clean up the temporary FASTA file
        os.remove(fasta_input)

        # Clean up the temporary output directory if it was created
        if output_dir.startswith(tempfile.gettempdir()):
            os.rmdir(output_dir)

    # Return the concatenated predicted secondary structure
    return predicted_structure

if __name__ == "__main__":
    # Example usage
    aa_sequence = "MVDDLYRTARPMYHEAELIREAGCPTICGVTLAN"
    predicted_structure = predict_secondary_structure(aa_sequence)
    print(predicted_structure)
