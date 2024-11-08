import subprocess
import os
import tempfile
import pandas as pd

def predict_secondary_structure_batch(aa_sequences, output_dir=None, device="gpu"):
    # If no output directory is provided, use a temporary directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp()

    # Create a FASTA-format string for all the amino acid sequences
    fasta_content = ""
    for idx, sequence in enumerate(aa_sequences):
        fasta_content += f">seq_{idx}\n{sequence}\n"

    # Create a temporary FASTA file to store the sequences
    with tempfile.NamedTemporaryFile(delete=False, suffix=".fasta") as temp_fasta:
        fasta_input = temp_fasta.name
        temp_fasta.write(fasta_content.encode())
    
    try:
        # Define the command with a custom prefix for the output file
        cmd = [
            "python3", "run_model.py",
            "--outfmt", "ss2",
            "--device", device,
            fasta_input
        ]

        # Capture the output using subprocess and extract only the secondary structures
        result = subprocess.run(cmd, cwd="/content/convergent_overlaps_aa_change/s4pred", 
                                capture_output=True, text=True, check=True)
        
        # Parse the output to match secondary structures with the input sequences
        secondary_structures = []
        current_structure = []
        for line in result.stdout.splitlines():
            #print(f"Processing line: {line}")  # Print the line being processed
            if line.startswith("#"):
                continue  # Skip comment lines
            if len(line.split()) < 3:
                if current_structure:
                    secondary_structures.append("".join(current_structure))
                    current_structure = []
                continue
            parts = line.split()
            current_structure.append(parts[2])  # Column 3 contains the secondary structure

        if current_structure:  # Add the last sequence's structure
            secondary_structures.append("".join(current_structure))

    finally:
        # Clean up the temporary FASTA file
        os.remove(fasta_input)

        # Clean up the temporary output directory if it was created
        if output_dir.startswith(tempfile.gettempdir()):
            os.rmdir(output_dir)

    # Return the list of predicted secondary structures
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
