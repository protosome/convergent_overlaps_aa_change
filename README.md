# Convergent Overlaps AA Change - README

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/protosome/convergent_overlaps_aa_change/blob/main/convergent_overlapping_gene_generation.ipynb)


## Project Overview
This project, **Convergent Overlaps AA Change**, is focused on analyzing overlapping amino acid sequences and predicting DNA back-translation. The core functionality involves training and using transformer encoder-based models to predict overlapping DNA sequences that encode two given amino acid sequences. The project also integrates secondary structure prediction using S4Pred.

## Getting Started
Follow these steps to get started with this project in Google Colab, which is currently the fastest way to get started. It can be run locally with a supported GPU, however currently this will require updates to file paths in the notebook as appropriate.

## Running the Notebook in Colab
Click on the "Open In Colab" badge at the top of this README to open the notebook in Google Colab and start running the code interactively.

### Clone the Repository
Clone the project repository by running the appropriate command to copy the project files to your local environment.

## Getting Started
Follow these steps to get started with this project in Google Colab:

### Clone the Repositories and Install Required Dependencies 
Clone the repository as well as other required dependencies (eg, biopython, fair-esm, and s4pred):

```bash
git clone https://github.com/protosome/convergent_overlaps_aa_change.git
cd /content/convergent_overlaps_aa_change/
pip install biopython
pip install fair-esm
cd /content/convergent_overlaps_aa_change/s4pred
wget http://bioinfadmin.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz
tar -xvzf weights.tar.gz
cd /content/convergent_overlaps_aa_change/
```

### Tokenization and Text Vectorization
To work with the sequences, you will need to tokenize and vectorize them. Tokenization is necessary to convert sequences into a numerical format that the model can understand, while vectorization ensures that all sequences are of consistent length for processing. This is done automatically with the provided tokenizers.

### Predict Overlapping Sequence for Two Amino Acid Sequences
The main functionality of this project is to predict overlapping DNA sequences for two given amino acid sequences. This is done using transformer-based models trained specifically for this purpose. An overlap length from 199 to 312 may be user specified when generating and optimizing the overlap.

### Sequence Inputs
An Excel (.xlsx) or CSV file containing two amino acid sequences (minimum length: 105 amino acids each) in the columns" "**aa_seq_1**" and "**aa_seq_2**". 

If desired, amino acids can be designated for preservation by placing square brackets around them. For example, given the sequence MRTSSRT, writing MR[TS]SRT will preserve the amino acids TS during overlap generation and optimization. Add these to columns "**aa_seq_1_brackets**" and "**aa_seq_2_brackets**". Note: if both sequences contain preserved residues at the same relative position, this may create conflicts and reduce the feasibility of generating valid overlaps.

If no sequence preservation is required, simply copy the same input sequence into the bracket columns.



## Repository Structure
- **analysis.ipynb**: Main notebook for running the analysis.
- **s4pred/**: Directory containing scripts and model weights for secondary structure prediction.
- **aa_change_model_set/**: Directory containing model data files.
- **protsub_matrix.py** and **blosum62_matrix.py**: Modules for similarity calculations.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **S4Pred**: For secondary structure prediction.
- **ESM-2**: for contact-map embeddings.
- **BioPython**: For sequence analysis and manipulation.

## Dependencies and Citations

This project makes use of the following external repositories and packages. If you use this work in academic research, please also cite the corresponding publications where applicable:

S4Pred (secondary structure prediction)
Repository: https://github.com/psipred/s4pred
Reference: Moffat L, Jones DT. Increasing the accuracy of single sequence prediction methods using a deep semi-supervised learning framework. Xu J, editor. Bioinformatics. 2021 Nov 5;37(21):3744–51.  

ESM-2 (Evolutionary Scale Modeling)
Repository: https://github.com/facebookresearch/esm and https://huggingface.co/facebook/esm2_t12_35M_UR50D (for models)
Reference: Lin Z, Akin H, Rao R, et al. Language models of protein sequences at the scale of evolution enable accurate structure prediction. Science. 2023 Mar 17;379(6637):1123–30.

Biopython
Repository: https://github.com/biopython/biopython
Reference: Cock PJA, Antao T, Chang JT, et al. Biopython: freely available Python tools for computational molecular biology and bioinformatics. Bioinformatics, 2009.

---
This README should provide you with the information you need to get started with the Convergent Overlaps AA Change project. If you encounter any issues or have questions, please feel free to reach out or open an issue on GitHub.
