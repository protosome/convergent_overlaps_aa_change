# Convergent Overlaps AA Change - README

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/protosome/convergent_overlaps_aa_change/blob/main/analysis.ipynb)

## Project Overview
This project, **Convergent Overlaps AA Change**, is focused on analyzing overlapping amino acid sequences and predicting DNA back-translation. The core functionality involves training and using transformer encoder-based models to predict overlapping DNA sequences that encode two given amino acid sequences. The project also integrates secondary structure prediction using S4Pred.

## Getting Started
Follow these steps to get started with this project in Google Colab:

### Clone the Repository
Clone the project repository by running the appropriate command to copy the project files to your local environment.

## Getting Started
Follow these steps to get started with this project in Google Colab:

### Clone the Repository
Clone the repository and move into the project directory:

git clone https://github.com/protosome/convergent_overlaps_aa_change.git
cd convergent_overlaps_aa_change

### Setup
Navigate to the required directory and download the necessary model weights. Extract the downloaded weights and navigate back to the project directory.

### Install Dependencies
Install the required dependencies for the project. This includes libraries for sequence analysis, model training, and deep learning.

## Usage
This project utilizes Python and several deep learning libraries. Below are the main components to get you started with using the provided models for sequence analysis.

### Tokenization and Text Vectorization
To work with the sequences, you will need to tokenize and vectorize them. Tokenization is necessary to convert sequences into a numerical format that the model can understand, while vectorization ensures that all sequences are of consistent length for processing.

### Predict Overlapping Sequence for Two Amino Acid Sequences
The main functionality of this project is to predict overlapping DNA sequences for two given amino acid sequences. This is done using transformer-based models trained specifically for this purpose.

## Running the Notebook in Colab
Click on the "Open In Colab" badge at the top of this README to open the notebook in Google Colab and start running the code interactively.

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

---
This README should provide you with the information you need to get started with the Convergent Overlaps AA Change project. If you encounter any issues or have questions, please feel free to reach out or open an issue on GitHub.
