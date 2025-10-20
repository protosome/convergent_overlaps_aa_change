# Designing Convergent Overlapping Genes - README

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/protosome/convergent_overlaps_aa_change/blob/main/convergent_overlapping_gene_generation.ipynb)

![Scrolling MSA](assets/msa_scroll_side_by_side.gif)

## Project Overview
This project, **Designing Convergent Overlapping Genes**, is focused on analyzing overlapping amino acid sequences and predicting DNA back-translation. The core functionality involves training and using transformer encoder-based models to predict overlapping DNA sequences that encode two given amino acid sequences. The multi-objective optimization process integrates secondary structure predictions using S4PRED, long-range contacts using ESM-2 contact maps, alignment scores, and substitution scores.

## Getting Started
Follow these steps to get started with this project in Google Colab, which is currently the fastest way to start predicting convergent overlaps. It can be run locally with a supported GPU, however currently this will require updates to file paths in the notebook as appropriate.

Of note, if running locally:

- Running inference is not generally limited by VRAM unless you opt for a large ESM-2 model for contact map generation.
- A modern consumer-grade NVIDIA GPU with a high number of CUDA cores will typically provide faster inference and overlap generation than the GPUs offered in Google Colab.

## Running the Notebook in Colab
Click on the "Open In Colab" badge at the top of this README to open the notebook in Google Colab and start running the code interactively.

### Clone the Repositories and Install Required Dependencies 
Clone the repository as well as other required dependencies (eg, biopython, fair-esm, and s4pred):

```bash
!git clone https://github.com/protosome/convergent_overlaps_aa_change.git
%cd convergent_overlaps_aa_change
from paths import ROOT_DIR
print("Root dir is:", ROOT_DIR)
!pip install biopython
!pip install fair-esm
%cd {ROOT_DIR}/s4pred
!wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz
!tar -xvzf weights.tar.gz
%cd {ROOT_DIR}
```

### Tokenization and Text Vectorization
To work with the sequences, you will need to tokenize and vectorize them. Tokenization is necessary to convert sequences into a numerical format that the model can understand, while vectorization ensures that all sequences are of consistent length for processing. This is done automatically with the provided tokenizers.

### Predict Overlapping Sequence for Two Amino Acid Sequences
The main functionality of this project is to predict overlapping DNA sequences for two given amino acid sequences. This is done using transformer-based models trained specifically for this purpose. An overlap length from 199 to 312 may be user specified when generating and optimizing the overlap.

### Input File Format

Provide an Excel (`.xlsx`) or CSV file with two amino acid sequences (minimum length: **105 amino acids each**).  

- **Required columns:**
  - `aa_seq_1`  
  - `aa_seq_2`

- **Optional preservation columns:**  
  You may mark specific amino acids to **preserve** during overlap generation by placing them in square brackets.  

  **Example:**  
  - Sequence: `MRTSSRT`  
  - Preserved form: `MR[TS]SRT` → amino acids **TS** will be preserved.  

  Add these bracketed sequences to:  
  - `aa_seq_1_brackets`  
  - `aa_seq_2_brackets`

> ⚠️ If both sequences specify preserved residues at the same relative position, this can create conflicts and reduce the chance of generating valid overlaps.  

If no preservation is required, you can simply **copy the same input sequences** into the bracket columns (or do not include the bracket columns in the input file).

---

### Demo Table

| aa_seq_1     | aa_seq_2     | aa_seq_1_brackets | aa_seq_2_brackets |
|--------------|--------------|-------------------|-------------------|
| MRTSSRT...   | QLGDVKP...   | MR[TS]SRT...      | QLGD[VK]P...      |
| AGPLMNQ...   | RTYKSDH...   | AGPLMNQ...        | RTYKSDH...        |

*(“...” indicates continuation; each sequence must be ≥105 amino acids long.)*

### Output
After running, one excel file will be generated per row from the input file. Each file will contain every amino acid pair from the windowed optimization process. To start, the recommendation is to sort for the higest value in the "combined score" column. However, given the stochastic nature of the optimization process, it is not safe to assume that the final sequence pair predicted will be the best.

From the output file, the "**translated_integrated_seq_1**" and "**translated_integrated_seq_2**" columns contain the full integrated amino acid sequences, and the "**integrated_seq_1**" and "**integrated_seq_2**" columns contain the integrated full DNA sequences for both amino acid sequences, respectively. These are the DNA sequences that include the convergent overlaps.

## Repository Structure
- **convergent_overlapping_gene_generation.ipynb**: Main notebook for running the analysis.
- **s4pred/**: Directory containing scripts and model weights for secondary structure prediction.
- **aa_change_model_set/**: Directory containing model data files.
- **protsub_matrix.py** and **blosum62_matrix.py**: Modules for similarity calculations.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **S4PRED**: For secondary structure prediction.
- **ESM-2**: For contact-map embeddings.
- **BioPython**: For sequence analysis and manipulation.

## Dependencies and Citations

This project makes use of the following external repositories and packages. If you use this work in academic research, please also cite the corresponding publications where applicable:

- S4PRED (secondary structure prediction). Repository: https://github.com/psipred/s4pred. Reference: Moffat L, Jones DT. Increasing the accuracy of single sequence prediction methods using a deep semi-supervised learning framework. Xu J, editor. Bioinformatics. 2021 Nov 5;37(21):3744–51.  

- ESM-2 (Evolutionary Scale Modeling). Repository: https://github.com/facebookresearch/esm and https://huggingface.co/facebook/esm2_t12_35M_UR50D (for models). Reference: Lin Z, Akin H, Rao R, et al. Language models of protein sequences at the scale of evolution enable accurate structure prediction. Science. 2023 Mar 17;379(6637):1123–30.

- Biopython. Repository: https://github.com/biopython/biopython. Reference: Cock PJA, Antao T, Chang JT, et al. Biopython: freely available Python tools for computational molecular biology and bioinformatics. Bioinformatics, 2009.

---
This README should provide you with the information you need to get started with the **Designing Convergent Overlapping Genes** project. If you encounter any issues or have questions, please feel free to reach out or open an issue on GitHub.
