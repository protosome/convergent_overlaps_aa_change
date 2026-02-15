"""
This module contains the dependency and core-function cells for notebook-free execution (while allowing for notebook execution in Colab).
"""

from __future__ import annotations

from pathlib import Path
from paths import ROOT_DIR

#@title ###**Load dependencies**.

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Seq import CodonTable
import itertools as it
import pandas as pd
import math
import random
import pickle
from protsub_matrix import prot_sub_matrix, calculate_protsub_similarity
from blosum62_matrix import blosum62_matrix, calculate_blosum62_similarity
from running_s4pred import predict_secondary_structure # This loads the S4PRED function to run as a subprocess, outputting only the structure prediction sequence
from running_s4pred_batch_fast import predict_secondary_structure_with_progress # Updated version of the S4PRED function, in faster batches
from transformer_encoder_model import TransformerModel, SinusoidalPositionalEncoding, TransformerBlock
import openpyxl
import time, math
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any, Iterable
from skimage.metrics import structural_similarity as ssim
import re
import io
import hashlib
import sys
from datetime import datetime
from contextlib import redirect_stdout


#@title ###**Main and supporting functions**.

# Identify available GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
torch.cuda.get_device_name(0) if torch.cuda.is_available() else print('CUDA not available; using CPU')

# Tokenization and text vectorization
max_length = 315  # max len of the overlap
vocab_size = 27

# Define a function to load the tokenizer
def load_tokenizer(filename):
    tok_path = Path(filename)
    if not tok_path.is_absolute():
        tok_path = ROOT_DIR / tok_path
    with open(tok_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

concat_tokenizer = load_tokenizer('concat_tokenizer.pkl')
overlap_tokenizer = load_tokenizer('overlap_tokenizer.pkl')

# Function to tokenize sentences using TensorFlow's tokenizer
def tokenize(sentences):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, filters='')
    tokenizer.fit_on_texts(sentences)  # Fit the tokenizer on the combined texts
    return tokenizer

# Function to vectorize sentences using the fitted tokenizer
def vectorize(tokenizer, sentences):
    seqs = tokenizer.texts_to_sequences(sentences)  # Convert texts to sequences of integers
    return tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=max_length, padding='post')  # Pad sequences

# Automatically test aa pairs in order, to identify those with matches
########################################################################################################################################

def select_and_update_sequences(
    final_results_df,
    sequence_1_aa_brackets,
    sequence_2_aa_brackets,
    score_threshold=None,
    avg_score_value=None,
    return_ranked=False
):
    # ---- filter ------------------------------------------------------
    filtered = final_results_df.copy()

    required = ['score_seq1', 'score_seq2', 'average_score_seq1_seq2']
    missing = [c for c in required if c not in final_results_df.columns]
    if missing:
        raise KeyError(f"select_and_update_sequences expected columns {missing} but they are not present.")

    if score_threshold is not None:
        mask = (
            filtered['score_seq1'].gt(score_threshold) &
            filtered['score_seq2'].gt(score_threshold)
        )
        filtered = filtered[mask]

    if avg_score_value is not None:
        filtered = filtered[
            filtered['average_score_seq1_seq2'] == avg_score_value
        ]

    if filtered.empty:                        # fall back to all rows
        filtered = final_results_df

    # ---- priority order ---------------------------------------------
    sorted_by_score = filtered.sort_values(
        by='average_score_seq1_seq2', ascending=False
    ).reset_index(drop=True)

    if return_ranked:
        return sorted_by_score

    # ---- legacy single-row return (unchanged) -----------------------
    best_row = sorted_by_score.iloc[0]
    upd1 = update_bracketed_sequence(
        best_row['translated_integrated_seq_1'], sequence_1_aa_brackets
    )
    upd2 = update_bracketed_sequence(
        best_row['translated_integrated_seq_2'], sequence_2_aa_brackets
    )
    return upd1, upd2

# ================================================================================================
# ---------------- FUNCTION: predict overlapping DNA sequence from two amino acid inputs ---------
# ================================================================================================


# Codons and their frequencies for each amino acid based on the E. coli table
back_translation_code_with_all_options = {
    'A': [('GCG', 0.27), ('GCT', 0.26), ('GCC', 0.26), ('GCA', 0.21)],
    'C': [('TGC', 0.53), ('TGT', 0.47)],
    'D': [('GAT', 0.63), ('GAC', 0.37)],
    'E': [('GAA', 0.68), ('GAG', 0.32)],
    'F': [('TTT', 0.58), ('TTC', 0.42)],
    'G': [('GGC', 0.35), ('GGT', 0.32), ('GGG', 0.25), ('GGA', 0.08)],
    'H': [('CAT', 0.56), ('CAC', 0.44)],
    'I': [('ATT', 0.48), ('ATC', 0.39), ('ATA', 0.14)],
    'K': [('AAA', 0.74), ('AAG', 0.26)],
    'L': [('CTG', 0.43), ('CTT', 0.13), ('CTC', 0.13), ('TTA', 0.14), ('CTA', 0.07), ('TTG', 0.13)],
    'M': [('ATG', 1.00)],
    'N': [('AAC', 0.60), ('AAT', 0.40)],
    'P': [('CCG', 0.52), ('CCA', 0.19), ('CCT', 0.16), ('CCC', 0.13)],
    'Q': [('CAG', 0.66), ('CAA', 0.34)],
    'R': [('CGT', 0.36), ('CGC', 0.36), ('CGG', 0.11), ('AGA', 0.08), ('AGG', 0.05), ('CGA', 0.04)],
    'S': [('AGC', 0.24), ('TCC', 0.24), ('TCT', 0.17), ('TCG', 0.15), ('TCA', 0.14), ('AGT', 0.15)],
    'T': [('ACC', 0.36), ('ACA', 0.28), ('ACG', 0.25), ('ACT', 0.11)],
    'V': [('GTG', 0.46), ('GTT', 0.28), ('GTC', 0.15), ('GTA', 0.11)],
    'W': [('TGG', 1.00)],
    'Y': [('TAT', 0.59), ('TAC', 0.41)],
    '*': [('TAA', 0.61), ('TGA', 0.30), ('TAG', 0.09)]
}

def predict_overlapping_sequence(trained_model, aa_sequences, concat_tokenizer, overlap_tokenizer):

    def translate_to_dna_with_all_options(aa_sequence: str) -> list:

        def choose_codon_based_on_frequency(codons):

            # Extract codon names and their frequencies
            codon_names = [codon for codon, _ in codons]
            codon_freqs = [freq for _, freq in codons]

            # Normalize the frequencies to ensure they sum up to 1
            total_frequency = sum(codon_freqs)
            normalized_freqs = [freq/total_frequency for freq in codon_freqs]

            # Randomly select a codon based on the frequency distribution
            chosen_codon = np.random.choice(codon_names, p = normalized_freqs)

            return [chosen_codon]

        # For each amino acid in the sequence, choose a codon based on its frequency
        list_of_list_of_codons = [choose_codon_based_on_frequency(back_translation_code_with_all_options[aa]) for aa in aa_sequence]

        # Combine the codons to form the nucleotide sequence
        list_of_combinations = [''.join(combination) for combination in it.product(*list_of_list_of_codons)]

        return list_of_combinations

    def translate(model, concat_tokenizer, overlap_tokenizer, text, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        vectorized_input = vectorize(concat_tokenizer, [text])
        vectorized_input = torch.tensor(vectorized_input, dtype=torch.long).to(device)

        # Perform inference
        with torch.inference_mode():
            output = model(vectorized_input)

        # Get the predicted classes
        _, predicted_classes = torch.max(output, dim=-1)
        predicted_classes = predicted_classes.cpu().numpy()

        # Function to reverse the predicted output back into text
        def decode_predictions(predicted_classes, tokenizer):
            # Get the index to word mapping from the tokenizer
            index_word = tokenizer.index_word

            # Convert indices to words
            predicted_text = ' '.join([index_word.get(index, '') for index in predicted_classes[0]])

            return predicted_text

        # Decode the predicted classes into text
        predicted_text = decode_predictions(predicted_classes, overlap_tokenizer)

        return predicted_text


    def convert_chars_for_translated_overlap(string):
        string = string.replace("no overlap", "#", 1)  # Replace first occurrence of "no overlap" with "#"
        string = string.replace("overlap", "#", 1)  # Replace first occurrence of "overlap" with "#"
        string = string.replace("no", "#", 1)  # Replace first occurrence of "no" with "#"
        if string == "#":
            return string
        new_string = ""
        for char in string:
            if char.lower() == "b":
                new_string += "A"
            elif char.lower() == "j":
                new_string += "G"
            elif char.lower() == "o":
                new_string += "T"
            elif char.lower() == "u":
                new_string += "C"
            else:
                new_string += char
        return new_string.replace(" ", "")


    # Running the above sequence through the model, using the tokenizer run on the data from the original model training process
    translated_overlap = translate(trained_model, concat_tokenizer, overlap_tokenizer, aa_sequences)

    overlap_output_from_model = convert_chars_for_translated_overlap(translated_overlap)

    # Generate the rc to include in the next small df
    reverse_complement_overlap = Seq(overlap_output_from_model)
    reverse_complement_overlap = str(reverse_complement_overlap.reverse_complement())

    # Create a DataFrame with two rows: original and reversed sequences
    # Note, the rc sequence actually comes first here, followed by the model output overlap sequence
    output_forward_reverse = pd.DataFrame({"overlap_sequence": [reverse_complement_overlap, overlap_output_from_model]})

    # Splitting and collapsing the sequences
    sequence_list = aa_sequences.split()
    collapsed_sequence = ''.join(sequence_list)

    # Splitting the sequence at the asterisk and keeping the asterisk
    split_sequences_with_asterisk = [seq + '*' for seq in collapsed_sequence.split('*') if seq]

    # Creating a dataframe from the sequences
    df_sequences = pd.DataFrame(split_sequences_with_asterisk, columns=["amino_acid_sequence"])

    # Adding the back translated dna sequences to the data frame.
    df_sequences['nt_sequence'] = df_sequences['amino_acid_sequence'].apply(lambda aa_seq: translate_to_dna_with_all_options(aa_seq)[0])

    df_sequences["overlap_seq"] = output_forward_reverse

    # Here, we are attaching the overlap sequence to the known coding sequence, generated above. It is
    # Added at the location based on the length of the overlap_seq
    def modify_sequence(df):
        # Check if 'overlap_seq' column exists in the DataFrame
        if 'overlap_seq' not in df.columns:
            raise ValueError("DataFrame must contain 'overlap_seq' column.")

        # Calculate the length of the overlap sequence
        df['overlap_length'] = df['overlap_seq'].apply(len)

        # Modify the sequences
        df['modified_sequence'] = df.apply(lambda row: row['nt_sequence'][:-row['overlap_length']] + row['overlap_seq'], axis=1)

        return df

    modify_sequence(df_sequences)

    # Custom function to translate using Seq
    def translate_with_seq(sequence):
        coding_dna = Seq(sequence)
        return str(coding_dna.translate())

    # Apply the custom function to the 'nt_sequence' column
    df_sequences['translated_sequence'] = df_sequences['modified_sequence'].apply(translate_with_seq)

    #print(compare_aa)
    return(df_sequences)


# Defining this align_sequences_identity function to check pariwise identity matching, since the globalxx approach allows sequence shifting,
# Which is problematic in the partial-matching predictions.

def align_sequences_identity(seq1, seq2):
    # Ensure sequences are of the same length
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length for 1:1 alignment")

    # Calculate the alignment score by comparing each character
    score = sum(1 for a, b in zip(seq1, seq2) if a == b)

    return score

# ================================================================================================
# ------------- FUNCTION: generate partially matching convergent overlap sequences --------------
# ================================================================================================

def find_partially_matching_sequence(model_to_predict, formatted_aa_seq, max_attempts, alignment_threshold_1, alignment_threshold_2, blosum_threshold_1, blosum_threshold_2, _counter=[0]):
    _counter[0] += 1

    print(f"Parsing row {_counter[0]}...")

    attempts = 0
    match_found = False
    matching_dataframe = None

    while attempts < max_attempts:
        attempts += 1

        try:
            result_df = predict_overlapping_sequence(model_to_predict, formatted_aa_seq, concat_tokenizer, overlap_tokenizer)
        except CodonTable.TranslationError as e:
            print(f"Error: {e}")
             # Print the "overlap_length" at the end of each iteration
            print(f"Attempt {attempts}... Overlap Length: {matching_dataframe['overlap_length'].iloc[0] if matching_dataframe is not None else 'N/A'}")
            continue

        # Check if the translated sequences contain more than just the terminal asterisk
        if result_df['translated_sequence'].iloc[0].count('*') > 1 or result_df['translated_sequence'].iloc[1].count('*') > 1:
            continue  # Skip the rest of the loop and try again if there are multiple asterisks

        # Calculate the new overlap length based on the provided conditions
        overlap_length = result_df['overlap_length'].iloc[0]

        if overlap_length % 3 == 0:
            overlap_length_3 = overlap_length / 3
        else:
            overlap_length_3 = (overlap_length + 1) / 3

        # Round the result to the nearest whole number
        overlap_length_3 = int(round(overlap_length_3, 0))

        # Truncate the sequences based on overlap_length_3
        truncated_aa_seq_1 = result_df['amino_acid_sequence'].iloc[0][-overlap_length_3:]
        truncated_trans_seq_1 = result_df['translated_sequence'].iloc[0][-overlap_length_3:]

        truncated_aa_seq_2 = result_df['amino_acid_sequence'].iloc[1][-overlap_length_3:]
        truncated_trans_seq_2 = result_df['translated_sequence'].iloc[1][-overlap_length_3:]

        blosum62_sim_score_1 = calculate_blosum62_similarity(truncated_aa_seq_1, truncated_trans_seq_1)
        blosum62_sim_score_2 = calculate_blosum62_similarity(truncated_aa_seq_2, truncated_trans_seq_2)

        blosum62_sim_score_1_ratio = (calculate_blosum62_similarity(truncated_aa_seq_1, truncated_trans_seq_1)/calculate_blosum62_similarity(truncated_aa_seq_1, truncated_aa_seq_1))
        blosum62_sim_score_2_ratio = (calculate_blosum62_similarity(truncated_aa_seq_2, truncated_trans_seq_2)/calculate_blosum62_similarity(truncated_aa_seq_2, truncated_aa_seq_2))

        protsub_sim_score_1 = calculate_protsub_similarity(truncated_aa_seq_1, truncated_trans_seq_1)
        protsub_sim_score_2 = calculate_protsub_similarity(truncated_aa_seq_2, truncated_trans_seq_2)

        # Perform new alignment on truncated sequences
        truncated_align_score_1 = align_sequences_identity(truncated_aa_seq_1, truncated_trans_seq_1) / overlap_length_3
        truncated_align_score_2 = align_sequences_identity(truncated_aa_seq_2, truncated_trans_seq_2) / overlap_length_3

        # Add new alignment scores to the DataFrame
        result_df['truncated_norm_align_score_1'] = round(truncated_align_score_1, 2)
        result_df['truncated_norm_align_score_2'] = round(truncated_align_score_2, 2)

        # Add blosum62 similarity scores to the DataFrame
        # Add new alignment scores to the DataFrame
        result_df['blosum62_sim_score_1'] = round(blosum62_sim_score_1, 2)
        result_df['blosum62_sim_score_1_len_norm'] = round(blosum62_sim_score_1 / overlap_length_3, 2)

        result_df['blosum62_sim_score_2'] = round(blosum62_sim_score_2, 2)
        result_df['blosum62_sim_score_2_len_norm'] = round(blosum62_sim_score_2 / overlap_length_3, 2)

        result_df['blosum62_sim_score_1_ratio'] = round(blosum62_sim_score_1_ratio, 2)
        result_df['blosum62_sim_score_2_ratio'] = round(blosum62_sim_score_2_ratio, 2)

        # Combine the two truncated alignment scores into one column
        result_df['blosum62_combined_scores'] = result_df.apply(
            lambda row: f"{row['blosum62_sim_score_1']} / {row['blosum62_sim_score_2']}",
            axis=1
        )

        # Combine the two truncated alignment scores into one column
        result_df['blosum62_combined_normalized_scores'] = result_df.apply(
            lambda row: f"{row['blosum62_sim_score_1_len_norm']} / {row['blosum62_sim_score_2_len_norm']}",
            axis=1
        )

        # Combine the two truncated alignment scores into one column
        result_df['blosum62_combined_ratio_scores'] = result_df.apply(
            lambda row: f"{row['blosum62_sim_score_1_ratio']} / {row['blosum62_sim_score_2_ratio']}",
            axis=1
        )

        # Combine the two truncated alignment scores into one column
        result_df['truncated_combined_scores'] = result_df.apply(
            lambda row: f"{row['truncated_norm_align_score_1']} / {row['truncated_norm_align_score_2']}",
            axis=1
        )

        # Add protsub similarity scores to the DataFrame
        # Add new alignment scores to the DataFrame
        result_df['protsub_sim_score_1'] = round(protsub_sim_score_1, 2)
        result_df['protsub_sim_score_1_len_norm'] = round(protsub_sim_score_1 / overlap_length_3, 2)

        result_df['protsub_sim_score_2'] = round(protsub_sim_score_2, 2)
        result_df['protsub_sim_score_2_len_norm'] = round(protsub_sim_score_2 / overlap_length_3, 2)

        # Combine the two truncated alignment scores into one column
        result_df['protsub_combined_scores'] = result_df.apply(
            lambda row: f"{row['protsub_sim_score_1']} / {row['protsub_sim_score_2']}",
            axis=1
        )

        # Combine the two truncated alignment scores into one column
        result_df['protsub_combined_normalized_scores'] = result_df.apply(
            lambda row: f"{row['protsub_sim_score_1_len_norm']} / {row['protsub_sim_score_2_len_norm']}",
            axis=1
        )

        # Set the alignment lower boundary desired
        align_limit_1 = alignment_threshold_1
        align_limit_2 = alignment_threshold_2
        blosum_limit_1 = blosum_threshold_1
        blosum_limit_2 = blosum_threshold_2

        if (truncated_align_score_1 >= align_limit_1 and
            truncated_align_score_2 >= align_limit_2 and
            result_df['blosum62_sim_score_1_len_norm'].iloc[0] >= blosum_limit_1 and
            result_df['blosum62_sim_score_2_len_norm'].iloc[0] >= blosum_limit_2 and
            result_df['translated_sequence'].iloc[0].count('*') == 1 and
            result_df['translated_sequence'].iloc[1].count('*') == 1 and
            result_df['translated_sequence'].iloc[0].endswith('*') and
            result_df['translated_sequence'].iloc[1].endswith('*')):
            match_found = True
            matching_dataframe = result_df
            break

    if match_found:
        # Print the combined truncated alignment scores
        combined_scores = matching_dataframe['truncated_combined_scores'].iloc[0]
        #combined_blosum_scores = matching_dataframe['blosum62_combined_scores'].iloc[0]
        combined_blosum_len_norm_scores = matching_dataframe['blosum62_combined_normalized_scores'].iloc[0]
        #combined_protsub_scores = matching_dataframe['protsub_combined_scores'].iloc[0]
        combined_protsub_len_norm_scores = matching_dataframe['protsub_combined_normalized_scores'].iloc[0]
        blosum62_combined_ratio_scores = matching_dataframe['blosum62_combined_ratio_scores'].iloc[0]
        print(f"Attempt {attempts}... Overlap Length: {matching_dataframe['overlap_length'].iloc[0] if matching_dataframe is not None else 'N/A'}")
        print(f"Truncated Combined Scores: {combined_scores}")
        #print(f"Blosum62 Similarity Scores: {combined_blosum_scores}")
        #print(f"Blosum62 Length Normalized Similarity Scores: {combined_blosum_len_norm_scores}")
        #print(f"Blosum62 Length Normalized Similarity Ratio Scores: {blosum62_combined_ratio_scores}")
        #print(f"ProtSub Similarity Scores: {combined_protsub_scores}")
        #print(f"ProtSub Length Normalized Similarity Scores: {combined_protsub_len_norm_scores}")
        return matching_dataframe

    else:
        print("There is no predicted significant overlap")
        return None


def process_sequences(aa_seq_1, aa_seq_2):

    if len(aa_seq_1) < 103:
        return "Error: sequence_1 is not long enough to process. The minimum length is 103 amino acids."
    if len(aa_seq_2) < 103:
        return "Error: sequence_2 is not long enough to process. The minimum length is 103 amino acids."

    # Keep only the final 104 amino acids of each sequence
    aa_seq_1_trimmed = aa_seq_1[-104:]
    aa_seq_2_trimmed = aa_seq_2[-104:]

    # Replace the first amino acid with an 'M'
    aa_seq_1_processed = 'M' + aa_seq_1_trimmed[1:]
    aa_seq_2_processed = 'M' + aa_seq_2_trimmed[1:]

    # Add an asterisk after each sequence
    aa_seq_1_processed += '*'
    aa_seq_2_processed += '*'

    # Concatenate the two sequences
    concatenated_seq = aa_seq_1_processed + aa_seq_2_processed

    # Add a space between every character
    final_seq = ' '.join(concatenated_seq)

    # Return the processed sequence
    return final_seq

def translate_to_dna_with_all_options(aa_sequence: str) -> str:

    # Codons and their frequencies for each amino acid based on the E. coli table
    back_translation_code_with_all_options = {
        'A': [('GCG', 0.27), ('GCT', 0.26), ('GCC', 0.26), ('GCA', 0.21)],
        'C': [('TGC', 0.53), ('TGT', 0.47)],
        'D': [('GAT', 0.63), ('GAC', 0.37)],
        'E': [('GAA', 0.68), ('GAG', 0.32)],
        'F': [('TTT', 0.58), ('TTC', 0.42)],
        'G': [('GGC', 0.35), ('GGT', 0.32), ('GGG', 0.25), ('GGA', 0.08)],
        'H': [('CAT', 0.56), ('CAC', 0.44)],
        'I': [('ATT', 0.48), ('ATC', 0.39), ('ATA', 0.14)],
        'K': [('AAA', 0.74), ('AAG', 0.26)],
        'L': [('CTG', 0.43), ('CTT', 0.13), ('CTC', 0.13), ('TTA', 0.14), ('CTA', 0.07), ('TTG', 0.13)],
        'M': [('ATG', 1.00)],
        'N': [('AAC', 0.60), ('AAT', 0.40)],
        'P': [('CCG', 0.52), ('CCA', 0.19), ('CCT', 0.16), ('CCC', 0.13)],
        'Q': [('CAG', 0.66), ('CAA', 0.34)],
        'R': [('CGT', 0.36), ('CGC', 0.36), ('CGG', 0.11), ('AGA', 0.08), ('AGG', 0.05), ('CGA', 0.04)],
        'S': [('AGC', 0.24), ('TCC', 0.24), ('TCT', 0.17), ('TCG', 0.15), ('TCA', 0.14), ('AGT', 0.15)],
        'T': [('ACC', 0.36), ('ACA', 0.28), ('ACG', 0.25), ('ACT', 0.11)],
        'V': [('GTG', 0.46), ('GTT', 0.28), ('GTC', 0.15), ('GTA', 0.11)],
        'W': [('TGG', 1.00)],
        'Y': [('TAT', 0.59), ('TAC', 0.41)],
        '*': [('TAA', 0.61), ('TGA', 0.30), ('TAG', 0.09)]
    }

    def choose_codon_based_on_frequency(codons):

        # Extract codon names and their frequencies
        codon_names = [codon for codon, _ in codons]
        codon_freqs = [freq for _, freq in codons]

        # Normalize the frequencies to ensure they sum up to 1
        total_frequency = sum(codon_freqs)
        normalized_freqs = [freq / total_frequency for freq in codon_freqs]

        # Randomly select a codon based on the frequency distribution
        chosen_codon = np.random.choice(codon_names, p=normalized_freqs)

        return chosen_codon

    # For each amino acid in the sequence, choose the most probable codon
    chosen_codons = [choose_codon_based_on_frequency(back_translation_code_with_all_options[aa]) for aa in aa_sequence]

    # Combine the chosen codons to form the nucleotide sequence
    nucleotide_sequence = ''.join(chosen_codons)

    return nucleotide_sequence

# Function to integrate modified sequence into original sequence
def integrate_modified_sequence(original_dna, modified_dna):
    # Remove the terminal xx nucleotides from the original sequence
    trimmed_original_dna = original_dna[:-309] # this should be max nt length, minus 6, since the backtranslated sequence does not have stop codon DNA seqs
    # Remove the first three nucleotides from the modified sequence
    trimmed_modified_dna = modified_dna[3:]
    # Integrate the modified sequence
    integrated_sequence = trimmed_original_dna + trimmed_modified_dna
    return integrated_sequence

def is_dna_sequence(sequence: str) -> bool:
    """
    Determine if a sequence is a DNA sequence.
    A DNA sequence should only contain A, T, C, G (and sometimes N).
    """
    return all(char in 'ATCGNatcgn' for char in sequence)

def load_model(model_path, vocab_size, embedding_dim, num_blocks, num_heads, ffn_dim, max_length, dropout_rate):
    # Define and load model architecture
    model = TransformerModel(
        vocab_size=vocab_size, embedding_dim=embedding_dim, num_blocks=num_blocks,
        num_heads=num_heads, ffn_dim=ffn_dim, max_length=max_length, dropout_rate=dropout_rate
    )
    model.load_state_dict(torch.load(model_path))
    model.train()
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


############################

def get_bracket_positions(sequence_with_brackets):
    """
    Extract positions of amino acids within brackets in sequence1 and map them to sequence2.

    Args:
        sequence_with_brackets (str): The amino acid sequence with bracketed sections.

    Returns:
        list of tuples: Each tuple contains the start position in sequence2 and the bracketed amino acids.
    """
    bracketed_amino_acids = []
    pos_with_brackets = 0  # Position in sequence1 (with brackets)
    pos_without_brackets = 0  # Position in sequence2 (without brackets)

    while pos_with_brackets < len(sequence_with_brackets):
        if sequence_with_brackets[pos_with_brackets] == '[':
            pos_with_brackets += 1  # Skip '['
            bracket_start_seq2 = pos_without_brackets  # Record start position in sequence2
            bracket_content = []

            # Collect all amino acids within the brackets
            while pos_with_brackets < len(sequence_with_brackets) and sequence_with_brackets[pos_with_brackets] != ']':
                bracket_content.append(sequence_with_brackets[pos_with_brackets])
                pos_with_brackets += 1
                pos_without_brackets += 1  # Advance sequence2 position as brackets are not present

            if pos_with_brackets >= len(sequence_with_brackets):
                raise ValueError("Unmatched '[' in sequence.")

            pos_with_brackets += 1  # Skip ']'
            bracketed_amino_acids.append((bracket_start_seq2, ''.join(bracket_content)))
        else:
            # Regular amino acid, advance both positions
            pos_with_brackets += 1
            pos_without_brackets += 1

    return bracketed_amino_acids

def compare_sequences_aa_selected(sequence1_with_brackets, sequence2):
    """
    Compare specific bracketed sections in sequence1 with sequence2.

    Args:
        sequence1_with_brackets (str): First amino acid sequence with brackets.
        sequence2 (str): Second amino acid sequence without brackets.

    Returns:
        dict: A dictionary with start positions in sequence2 as keys and True/False for match/mismatch.
    """
    bracketed_amino_acids = get_bracket_positions(sequence1_with_brackets)
    results = {}

    for start, amino_acids in bracketed_amino_acids:
        # Extract amino acids from sequence2 for the corresponding positions
        seq2_amino_acids = sequence2[start:start + len(amino_acids)]
        results[start] = (seq2_amino_acids == amino_acids)  # True if match, False if not

    return results

# Function to get comparison results between sequence1 and sequence2
def get_comparison_results(sequence1, sequence2):
    comparison_results = compare_sequences_aa_selected(sequence1, sequence2)
    return comparison_results

def match_status(comparison_results):
    """
    Determines the match status based on the values in comparison_results.

    Args:
        comparison_results (dict): A dictionary with match results (True/False values).

    Returns:
        str: A message indicating whether all, some, or none of the comparisons matched.
    """
    if all(comparison_results.values()):
        return "Match"
    elif any(comparison_results.values()):
        return "Partial match"
    else:
        return "No match"


# New version that only accounts for overlap secondary structure predictions for the comparison values
def compare_sequences(seq1, seq2, pred1, pred2, overlap_length):
    """
    Compare two sequences with their respective predictions and score the matches.

    Args:
    - seq1: Original sequence 1 (string)
    - seq2: Original sequence 2 (string)
    - pred1: Predicted sequence 1 (string)
    - pred2: Predicted sequence 2 (string)
    - overlap_length: Overlap length (int)

    Returns:
    - match_count: Number of positions where both sequences match their predictions
    - combined_score: Percentage score of combined matches relative to total positions
    - match_count_seq1: Number of positions where seq1 matches pred1
    - score_seq1: Percentage score of matches for seq1 relative to total positions
    - match_count_seq2: Number of positions where seq2 matches pred2
    - score_seq2: Percentage score of matches for seq2 relative to total positions
    - average_score_seq1_seq2: Mean average score of seq1 and seq2
    - abs_value: Absolute difference between score_seq1 and score_seq2
    """

    if overlap_length % 3 == 0:
        overlap_length_3 = overlap_length / 3
    else:
        overlap_length_3 = (overlap_length + 1) / 3

    # Round the result to the nearest whole number
    overlap_length_3 = int(round(overlap_length_3, 0))

    # Truncate the sequences based on overlap_length_3
    truncated_seq1 = seq1[-overlap_length_3:]
    truncated_seq2 = seq2[-overlap_length_3:]

    truncated_pred1 = pred1[-overlap_length_3:]
    truncated_pred2 = pred2[-overlap_length_3:]

    # Initialize counters
    total_positions = len(truncated_seq1)  # Assuming both sequences are the same length
    match_count = 0
    match_count_seq1 = 0
    match_count_seq2 = 0

    # Comparison loop
    for i in range(total_positions):
        if truncated_seq1[i] == truncated_pred1[i]:
            match_count_seq1 += 1
        if truncated_seq2[i] == truncated_pred2[i]:
            match_count_seq2 += 1
        if truncated_seq1[i] == truncated_pred2[i] and truncated_seq2[i] == truncated_pred2[i]:
            match_count += 1

    # Calculate scores
    combined_score = round(match_count / total_positions * 100, 2)
    score_seq1 = round(match_count_seq1 / total_positions * 100, 2)
    score_seq2 = round(match_count_seq2 / total_positions * 100, 2)
    average_score_seq1_seq2 = round((score_seq1 + score_seq2) / 2, 2)
    abs_value = round(abs(score_seq1 - score_seq2), 2)

    return (match_count, combined_score, match_count_seq1, score_seq1,
            match_count_seq2, score_seq2, average_score_seq1_seq2, abs_value)

# ================================================================================================
# ---------- HELPER: gather bracketed regions and compute per-region identity --------------------
# ================================================================================================

def bracket_alignment_stats(seq_with_brackets: str,
                            candidate_seq: str) -> tuple[int, float]:
    """
    Return (match_count, match_fraction) for the AA positions enclosed by [].

    Parameters
    ----------
    seq_with_brackets : str
        Reference sequence containing one or more bracketed regions,
        e.g. "...TT[T]L[TYG]V...".
    candidate_seq : str
        Sequence to compare against (no brackets).

    Returns
    -------
    match_count : int
        Number of positions in bracketed regions whose residues match `candidate_seq`.
    match_fraction : float
        match_count / total_bracket_length, rounded to 4 decimals.
        Returns 0.0 if there are zero bracketed positions (shouldn’t happen).
    """
    bracketed_regions = get_bracket_positions(seq_with_brackets)  # [(start, "AAA"), ...]
    total_len, match_count = 0, 0

    for start, ref_subseq in bracketed_regions:
        total_len += len(ref_subseq)
        # Compare residue‑by‑residue within this bracketed block
        for offset, ref_aa in enumerate(ref_subseq):
            if start + offset >= len(candidate_seq):
                raise IndexError(
                    f"Candidate sequence too short for bracket at pos {start}"
                )
            if candidate_seq[start + offset] == ref_aa:
                match_count += 1

    match_fraction = round(match_count / total_len, 4) if total_len else 0.0
    return match_count, match_fraction

# ================================================================================================
# --------- FUNCTIONS: predict secondary structures with S4PRED (batched processing) -------------
# ================================================================================================

output_dir = "{ROOT_DIR}/s4pred/outputs"

# This is used for single AA sequence secondary structure predictions
def structure_prediction_wrapper(sequence):
    output_dir = "{ROOT_DIR}/s4pred/outputs"

    sequence = sequence.replace('*', '')

    # Call the prediction function
    prediction = predict_secondary_structure(sequence, output_dir)

    # Print the sequence and its prediction
    print(f"Sequence: {sequence}")
    print(f"Predicted Structure: {prediction}")
    print("-" * 50)  # Separator for readability

    return prediction

def batch_structure_prediction_wrapper(sequences):
    # Normalize: wrap single string in a list
    if isinstance(sequences, str):
        sequences = [sequences]

    # Clean sequences
    cleaned_sequences = [seq.replace('*', '') for seq in sequences]

    # Always use the batched runner, even for 1 sequence
    predictions = predict_secondary_structure_with_progress(cleaned_sequences)
    return predictions

# ================================================================================================
# ---- FUNCTION: re-calc alignment score using original seq vs. re-run predicted seq (fine-tune) -
# ================================================================================================

# Function to define the alignment score if rerun is True.
def rerun_alignment_score(original_sequence, predicted_sequence, overlap_length):
    # Remove terminal asterisk if it exists
    if original_sequence.endswith('*'):
        original_sequence = original_sequence[:-1]
    if predicted_sequence.endswith('*'):
        predicted_sequence = predicted_sequence[:-1]

    # Calculate overlap length in amino acids
    if overlap_length % 3 == 0:
        overlap_length_3 = overlap_length / 3
    else:
        overlap_length_3 = (overlap_length + 1) / 3

    # Round the result to the nearest whole number
    overlap_length_3 = int(round(overlap_length_3, 0))

    # Truncate the sequences based on overlap_length_3
    truncated_aa_seq = original_sequence[-overlap_length_3:]
    truncated_trans_seq = predicted_sequence[-overlap_length_3:]

    # Perform new alignment on truncated sequences
    truncated_align_score = align_sequences_identity(truncated_aa_seq, truncated_trans_seq) / overlap_length_3

    return truncated_align_score

# ================================================================================================
# -------- FUNCTION: split input sequence into two sequences (remove asterisk) -------------------
# ================================================================================================

def split_sequence(sequence):
    # Remove spaces from the sequence
    cleaned_sequence = sequence.replace(" ", "")

    # Split the sequence at asterisks and remove any empty strings in case of multiple asterisks
    split_seqs = [seq for seq in cleaned_sequence.split('*') if seq]

    # Ensure only two sequences are returned
    if len(split_seqs) == 2:
        return split_seqs[0], split_seqs[1]
    else:
        raise ValueError("The sequence does not split cleanly into two parts with a single asterisk.")

# ================================================================================================
# --------------------------- AUTOUPDATE: selected amino acids -----------------------------------
# ================================================================================================

def update_bracketed_sequence(seq, reference_bracketed_seq):
    """
    Replace the amino acids in 'seq' at the positions defined by
    the bracketed regions in 'reference_bracketed_seq' with the bracketed amino acids.

    Args:
        seq (str): The candidate sequence to be updated.
        reference_bracketed_seq (str): The reference sequence that includes bracketed amino acids.

    Returns:
        str: The updated sequence.
    """
    # Get the positions and the bracketed amino acids from the reference
    bracket_positions = get_bracket_positions(reference_bracketed_seq)
    # Convert the sequence into a mutable list
    seq_list = list(seq)
    for pos, aa in bracket_positions:
        # Replace the substring with the bracketed amino acids.
        # (Assumes that the positions match the intended residues.)
        seq_list[pos: pos + len(aa)] = list(aa)
    return "".join(seq_list)

# ================================================================================================
# ----------------------------------- MODEL DATA FILE --------------------------------------------
# ================================================================================================

# This is the 199 to 312 models set, with aa changes.
model_data = pd.read_excel('aa_change_models/overlap_length_models_aa_change_random_pairs_315nt_length_199_312_20250326_colab.xlsx')

# ================================================================================================
# --------- AUTO-TEST: amino acid pairs in order to identify those with matches ------------------
# ================================================================================================

def select_and_update_sequences(
    final_results_df,
    sequence_1_aa_brackets,
    sequence_2_aa_brackets,
    score_threshold=None,
    avg_score_value=None,
    return_ranked=False
):
    # ---- filter ------------------------------------------------------
    filtered = final_results_df.copy()

    required = ['score_seq1', 'score_seq2', 'average_score_seq1_seq2']
    missing = [c for c in required if c not in final_results_df.columns]
    if missing:
        raise KeyError(f"select_and_update_sequences expected columns {missing} but they are not present.")

    if score_threshold is not None:
        mask = (
            filtered['score_seq1'].gt(score_threshold) &
            filtered['score_seq2'].gt(score_threshold)
        )
        filtered = filtered[mask]

    if avg_score_value is not None:
        filtered = filtered[
            filtered['average_score_seq1_seq2'] == avg_score_value
        ]

    if filtered.empty:                        # Fall back to all rows
        filtered = final_results_df

    # ---- priority order ---------------------------------------------
    sorted_by_score = filtered.sort_values(
        by='average_score_seq1_seq2', ascending=False
    ).reset_index(drop=True)

    if return_ranked:
        return sorted_by_score

    # ---- Legacy single-row return -----------------------
    best_row = sorted_by_score.iloc[0]
    upd1 = update_bracketed_sequence(
        best_row['translated_integrated_seq_1'], sequence_1_aa_brackets
    )
    upd2 = update_bracketed_sequence(
        best_row['translated_integrated_seq_2'], sequence_2_aa_brackets
    )
    return upd1, upd2

# ================================================================================================
# ----------------------------- RANGE MERGER (deduplicates / coalesces) --------------------------
# ================================================================================================

def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merge overlapping or immediately adjacent (start, end) index ranges.

    Parameters
    ----------
    ranges : List[Tuple[int, int]]
        Inclusive index pairs, e.g. [(12, 37), (35, 50)].

    Returns
    -------
    List[Tuple[int, int]]
        Sorted, non‑overlapping index pairs.
    """
    if not ranges:
        return []

    # Sort by start index, then sweep‑merge
    ranges = sorted(ranges, key=lambda r: r[0])
    merged = [list(ranges[0])]

    for s, e in ranges[1:]:
        # If current range overlaps or kisses the previous one, extend it
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # Cast inner lists back to tuples
    return [tuple(r) for r in merged]

# ================================================================================================
# ----------------------------- FC‑DROPOUT RANGE BUILDER -----------------------------------------
# ================================================================================================

def build_fc_dropout_ranges(seq_bracketed: str,
                            window_aa: int = 105,
                            tok_len: int = 315,
                            invert: bool = False) -> List[Tuple[int, int]]:
    """
    Convert bracketed amino‑acid notation into nucleotide‑token index ranges to be
    used for FC‑layer token‑dropout.

    Parameters
    ----------
    seq_bracketed : str
        Full amino‑acid sequence with regions to protect / measure enclosed in '[]'.
    window_aa     : int
        Length (in amino acids) of the C‑terminal slice that is fed to the model.
    tok_len       : int
        Length of the nucleotide token axis (== window_aa * 3).
    invert        : bool, default False
        Set True for sequences that are used in *reverse* orientation (seq‑2),
        producing indices relative to the non‑flipped mask.

    Returns
    -------
    List[Tuple[int, int]]
        Sorted, non‑overlapping (start, end) pairs **inclusive** on the token axis.
    """
    # ------------------------------------------------------------------
    # 1) Locate every AA index that appears inside brackets
    # ------------------------------------------------------------------

    aa_idx = -1            # Running index across *true* amino‑acid letters
    in_bracket = False
    current = []
    groups: List[List[int]] = []

    for ch in seq_bracketed:
        if ch == '[':
            in_bracket = True
            continue
        if ch == ']':
            in_bracket = False
            if current:
                groups.append(current)
                current = []
            continue
        if not ch.isalpha():   # Ignore spaces / punctuation
            continue

        aa_idx += 1
        if in_bracket:
            current.append(aa_idx)

    if current:                 # Handles a bracket that reaches the last char
        groups.append(current)

    aa_len = aa_idx + 1
    if aa_len < window_aa:
        raise ValueError("window_aa exceeds available AA length")

    # ------------------------------------------------------------------
    # 2) Map AA indices -> nucleotide‑token indices (0 … tok_len‑1)
    #    token 0  == last nucleotide (3' end)
    # ------------------------------------------------------------------

    ranges: List[Tuple[int, int]] = []
    for g in groups:
        # Discard any AA that lies *outside* the final window_aa slice
        valid = [pos for pos in g if (aa_len - 1 - pos) < window_aa]
        if not valid:
            continue

        # Contiguous by construction – only need first & last
        first, last = valid[0], valid[-1]

        # Distance from C‑terminus (k = 0 is very last AA)
        k_start = aa_len - 1 - last      # closest to 3' end
        k_end   = aa_len - 1 - first     # farthest within window

        tok_start = k_start * 3
        tok_end   = k_end   * 3 + 2      # Inclusive

        if invert:                       # reverse orientation (seq‑2)
            tok_start, tok_end = tok_len - 1 - tok_end, tok_len - 1 - tok_start

        ranges.append((tok_start, tok_end))

    # ------------------------------------------------------------------
    # 3) Merge any overlaps and sort
    # ------------------------------------------------------------------

    ranges.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in ranges:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    return [tuple(r) for r in merged]


# ================================================================================================
# ----------------------------- FC DROPOUT / CONSTRAINT UTILITIES --------------------------------
# ================================================================================================

def _build_bool_mask_from_ranges(L: int,
                                 ranges: Optional[List[Tuple[int, int]]]) -> Optional[torch.Tensor]:
    if not ranges:
        return None
    m = torch.zeros(L, dtype=torch.bool)
    for s, e in ranges:
        if s < 0 or e < 0 or s >= L or e >= L or e < s:
            raise ValueError(f"Bad range ({s},{e}) for length {L}")
        m[s:e+1] = True
    return m


# Mapping from dictionary and char→nucleotide rules:
NUC_TO_CLASS: Dict[str, int] = {'T': 1, 'A': 2, 'C': 3, 'G': 4}
CLASS_TO_NUC: Dict[int, str] = {v: k for k, v in NUC_TO_CLASS.items()}

def classes_from_nuc_string(nuc_string: str) -> List[int]:
    """Map a nucleotide string like 'TACG' to class indices [1,2,3,4]."""
    out: List[int] = []
    for ch in nuc_string:
        ch = ch.upper()
        if ch in NUC_TO_CLASS:
            out.append(NUC_TO_CLASS[ch])
    return out

def build_fixed_logits_spec(positions: List[int],
                            classes: List[int],
                            L: int) -> Dict[int, int]:
    """Return {pos: class_idx} with bounds checks."""
    if len(positions) != len(classes):
        raise ValueError("positions and classes must be same length")
    spec: Dict[int, int] = {}
    for p, c in zip(positions, classes):
        if p < 0 or p >= L:
            raise ValueError(f"Bad fixed position {p} for length {L}")
        spec[p] = int(c)
    return spec

def fixed_spec_from_nuc_string(nuc_string: str,
                               start_pos: int,
                               tok_len: int) -> Dict[int, int]:
    """Convenience: 'TACG', start_pos=100 -> {100:1,101:2,102:3,103:4}."""
    cls = classes_from_nuc_string(nuc_string)
    positions = list(range(start_pos, start_pos + len(cls)))
    return build_fixed_logits_spec(positions, cls, L=tok_len)


def _apply_fixed_logits(out: torch.Tensor,
                        fixed_spec: Dict[int, int],
                        tok_ax: int,
                        fixed_value: float = 12.0,
                        blend_alpha: Optional[float] = None) -> torch.Tensor:
    """
    Enforce logits at positions in fixed_spec.

    out: logits tensor with class axis last; token axis = tok_ax
    fixed_value:
      - blend_alpha is None  -> hard one-hot: target=+fixed_value, others=-inf
      - blend_alpha in (0,1] -> soft blend: out=(1-α)·out + α·one_hot·fixed_value
    """
    # Move token axis to front for easy indexing
    perm = [tok_ax] + [i for i in range(out.dim()) if i != tok_ax]
    inv  = [perm.index(i) for i in range(len(perm))]
    x = out.permute(*perm)  # (T, ..., V)
    T = x.shape[0]

    for pos, cls in fixed_spec.items():
        if pos < 0 or pos >= T:
            continue
        sl = x[pos]  # (..., V)
        if blend_alpha is None:
            sl.fill_(-1e9)
            sl[..., cls] = fixed_value
        else:
            one_hot = torch.zeros_like(sl)
            one_hot[..., cls] = fixed_value
            sl.mul_(1.0 - blend_alpha).add_(one_hot, alpha=blend_alpha)

    return x.permute(*inv)


def _masked_logit_dropout_with_constraints(
    logits: torch.Tensor,
    token_mask_1d: Optional[torch.Tensor],
    p: float,
    training_flag: bool,
    tok_len: int = 315,
    fixed_logits_spec: Optional[Dict[int, int]] = None,
    fixed_value: float = 12.0,
    blend_alpha: Optional[float] = None,
    debug: bool = False
) -> torch.Tensor:
    # Identify token axis exactly like before
    if logits.shape[0] == tok_len:
        tok_ax = 0
    elif logits.ndim > 1 and logits.shape[1] == tok_len:
        tok_ax = 1
    else:
        # No token axis found; still allow fixed-only path
        if fixed_logits_spec:
            return logits  # or raise, but keeping old behavior
        return logits

    # If no dropout to apply, still enforce fixed logits if present
    if (not training_flag) or p == 0.0 or token_mask_1d is None:
        if fixed_logits_spec:
            return _apply_fixed_logits(
                logits, fixed_spec=fixed_logits_spec, tok_ax=tok_ax,
                fixed_value=fixed_value, blend_alpha=blend_alpha
            )
        return logits

    # Permute to put token axis first
    perm = [tok_ax] + [i for i in range(logits.dim()) if i != tok_ax]
    inv  = [perm.index(i) for i in range(len(perm))]
    x = logits.permute(*perm)  # (T, ..., V)
    T = x.shape[0]

    # Build/broadcast mask EXACTLY like the old function (per-class broadcasting)
    mask = token_mask_1d.to(x.device)
    if mask.shape[0] != T:
        raise ValueError(f"token_mask_1d length {mask.shape[0]} != token axis {T}")
    mask_view = mask.view(T, *([1] * (x.dim() - 1))).expand_as(x)  # note: -1 (not -2), covers class axis too

    # Do-not-drop mask for fixed positions, also broadcast over classes
    if fixed_logits_spec:
        fixed_mask_t = torch.zeros(T, dtype=torch.bool, device=x.device)
        for pos in fixed_logits_spec.keys():
            if 0 <= pos < T:
                fixed_mask_t[pos] = True
        fixed_mask_view = fixed_mask_t.view(T, *([1] * (x.dim() - 1))).expand_as(x)
    else:
        fixed_mask_view = torch.zeros_like(x, dtype=torch.bool)

    # Per-class dropout
    rand = torch.rand_like(x)
    drop = (rand < p) & mask_view & (~fixed_mask_view)
    keep = ~drop

    out = x * keep.to(x.dtype)

    # Inverted dropout scaling on all masked, non-fixed entries (matches old)
    scale = torch.ones_like(x)
    scale[mask_view & (~fixed_mask_view)] = 1.0 / (1.0 - p)
    out = out * scale

    # Now enforce fixed logits (hard/soft) at those positions
    if fixed_logits_spec:
        out = _apply_fixed_logits(out, fixed_spec=fixed_logits_spec, tok_ax=0,
                                  fixed_value=fixed_value, blend_alpha=blend_alpha)

    return out.permute(*inv)

def _find_output_linear_single(model: nn.Module,
                               vocab_size_guess: Optional[int] = None):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return [("fc", model.fc)]
    cands = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not cands:
        return []
    if vocab_size_guess is not None:
        vs = [(n, m) for (n, m) in cands if m.out_features == vocab_size_guess]
        if vs:
            return [vs[-1]]
    return [cands[-1]]

def install_fc_token_dropout_and_constraints_hook(model: nn.Module,
                                                  token_mask_1d: Optional[torch.Tensor],
                                                  p: float,
                                                  force_eval: bool,
                                                  tok_len: int,
                                                  vocab_size_guess: Optional[int] = None,
                                                  fixed_logits_spec: Optional[Dict[int, int]] = None,
                                                  fixed_value: float = 12.0,
                                                  blend_alpha: Optional[float] = None,
                                                  debug: bool = False):
    """
    Registers a forward hook on the final Linear to apply:
      - masked token dropout (MC) in the selected region
      - fixed logits at user-specified token positions

    If you want only fixed logits (no dropout), pass p=0; token_mask_1d may be None.
    """
    # Still set up the hook if fixed logits are requested even without a mask/p>0
    if (p <= 0.0 and not fixed_logits_spec) and token_mask_1d is None:
        return []

    targets = _find_output_linear_single(model, vocab_size_guess)
    if not targets:
        return []

    def make_hook():
        def hook(_module, _inp, out):
            training_flag = _module.training or force_eval
            return _masked_logit_dropout_with_constraints(
                out,
                token_mask_1d=token_mask_1d,
                p=p,
                training_flag=training_flag,
                tok_len=tok_len,
                fixed_logits_spec=fixed_logits_spec,
                fixed_value=fixed_value,
                blend_alpha=blend_alpha,
                debug=debug
            )
        return hook

    handles = []
    for _name, lin in targets:
        h = lin.register_forward_hook(make_hook())
        handles.append(h)
    return handles

def shift_and_mirror_fixed_logits_positions_forward(
    fixed_spec: Dict[int, int],
    tok_len: int,
    model_number: int
) -> Dict[int, int]:
    """
    Shift fixed logits positions forward, then mirror across tok_len to align with reverse predictions.
    Excludes any positions that fall outside [0, tok_len-1].
    """
    shift = 0  # Change if offset is desired, but not recommended

    mirrored = {}
    for pos, cls in fixed_spec.items():
        # Apply shift
        shifted_pos = pos - shift

        # Skip if shifted position is invalid
        if shifted_pos < 0 or shifted_pos >= tok_len:
            continue

        # Mirror and skip if mirrored position is invalid
        mirrored_pos = (tok_len - 1) - shifted_pos
        if mirrored_pos < 0 or mirrored_pos >= tok_len:
            continue

        mirrored[mirrored_pos] = cls

    return mirrored

def swap_fixed_logits_spec_classes_only_reverse(
    fixed_spec: Dict[int, int],
    tok_len: int,
    model_number: int
) -> Dict[int, int]:
    """
    Swap nucleotide class IDs only (no position mirroring) and shift positions:
      - Shift each position by subtracting (tok_len - model_number).
      - Swap classes: 1 <-> 2, 3 <-> 4.
      - Other IDs remain unchanged.
    """
    swap_map = {1: 2, 2: 1, 3: 4, 4: 3}
    shift = tok_len - model_number
    swapped = {}
    for pos, cls in fixed_spec.items():
        new_pos = pos - shift
        swapped_cls = swap_map.get(cls, cls)
        swapped[new_pos] = swapped_cls
    return swapped

# ================================================================================================
# ----------------------------- MAIN INFERENCE FUNCTION ------------------------------------------
# ================================================================================================

def run_inference_for_models(
        model_numbers,
        seq_1,
        seq_2,
        max_attempts_first_pass,
        max_attempts_second_pass,
        first_pass_alignment_threshold_1,
        first_pass_alignment_threshold_2,
        second_pass_alignment_threshold_1,
        second_pass_alignment_threshold_2,
        first_pass_blosum_threshold_1,
        first_pass_blosum_threshold_2,
        second_pass_blosum_threshold_1,
        second_pass_blosum_threshold_2,
        first_pass_iterations,
        second_pass_iterations,
        inference_mode,
        feedforward_dropout_rate,
        attention_dropout_rate,
        set_seed,
        fc_dropout_p: float = 0.0,
        fc_ranges_forward: Optional[List[Tuple[int,int]]] = None,
        fc_ranges_reverse: Optional[List[Tuple[int,int]]] = None,
        tok_len: int = 315,
        apply_fc_dropout_in_eval: bool = True,
        vocab_size_guess: Optional[int] = None,
        fixed_logits_spec_forward: Optional[Dict[int, int]] = None,
        fixed_logits_spec_reverse: Optional[Dict[int, int]] = None,
        fixed_value: float = 12.0,
        blend_alpha: Optional[float] = None
    ):

    """
    Run inference for trained overlap models on a given
    amino acid sequence pair, with support for multi-pass optimization,
    dropout perturbations, and optional fixed-token enforcement.

    Parameters
    ----------
    model_numbers : list[int]
        List of model IDs / overlap lengths to run inference on.
    seq_1 : str
        First amino acid sequence.
    seq_2 : str
        Second amino acid sequence.

    # --- Retry / attempt control ---
    max_attempts_first_pass : int
        Number of retries allowed during the first-pass inference stage.
    max_attempts_second_pass : int
        Number of retries allowed during the second-pass refinement stage.

    # --- Alignment thresholds ---
    first_pass_alignment_threshold_1 : float
        Minimum alignment score for seq_1 during first-pass filtering.
    first_pass_alignment_threshold_2 : float
        Minimum alignment score for seq_2 during first-pass filtering.
    second_pass_alignment_threshold_1 : float
        Minimum alignment score for seq_1 during second-pass filtering.
    second_pass_alignment_threshold_2 : float
        Minimum alignment score for seq_2 during second-pass filtering.

    # --- BLOSUM thresholds ---
    first_pass_blosum_threshold_1 : float
        Minimum substitution (BLOSUM62) score for seq_1 in first pass.
    first_pass_blosum_threshold_2 : float
        Minimum substitution (BLOSUM62) score for seq_2 in first pass.
    second_pass_blosum_threshold_1 : float
        Minimum substitution score for seq_1 in second pass.
    second_pass_blosum_threshold_2 : float
        Minimum substitution score for seq_2 in second pass.

    # --- Iterations per stage ---
    first_pass_iterations : int
        Number of optimization iterations during first pass.
    second_pass_iterations : int
        Number of optimization iterations during second pass.

    # --- Inference mode and stochasticity ---
    inference_mode : str
        Mode selector (e.g. "deterministic", "stochastic", "mc_dropout").
    feedforward_dropout_rate : float
        Dropout probability applied in feedforward layers.
    attention_dropout_rate : float
        Dropout probability applied in attention layers.
    set_seed : Optional[int]
        Random seed to ensure reproducibility; None disables seeding.

    # --- Fine-grained dropout perturbations ---
    fc_dropout_p : float, default=0.0
        Probability of dropout applied in final classification (FC) layer.
    fc_ranges_forward : list[tuple[int,int]], optional
        Index ranges in the forward sequence where FC dropout is applied.
    fc_ranges_reverse : list[tuple[int,int]], optional
        Index ranges in the reverse sequence where FC dropout is applied.
    apply_fc_dropout_in_eval : bool, default=True
        If True, apply FC dropout during evaluation/inference (MC Dropout).

    # --- Sequence / tokenization control ---
    tok_len : int, default=315
        Token length (padded/truncated length of overlap sequences).
    vocab_size_guess : int, optional
        Override for vocabulary size if not inferable from tokenizer.

    # --- Fixed logits overrides ---
    fixed_logits_spec_forward : dict[int,int], optional
        Dictionary mapping token positions → fixed AA index for seq_1.
    fixed_logits_spec_reverse : dict[int,int], optional
        Dictionary mapping token positions → fixed AA index for seq_2.
    fixed_value : float, default=12.0
        Logit strength assigned to enforced fixed tokens.

    # --- Blending controls ---
    blend_alpha : float, optional
        Weight (0–1) for blending logits from two candidate predictions.

    Returns
    -------
    results : dict
        Dictionary containing inference outputs, scores, and metadata.
    """

    if set_seed is None:
        torch.manual_seed(int(time.time()))
    else:
        torch.manual_seed(set_seed)

    if is_dna_sequence(seq_1) and is_dna_sequence(seq_2):
        aa_seq_1 = str(Seq(seq_1).translate())[:-1]
        aa_seq_2 = str(Seq(seq_2).translate())[:-1]
    else:
        aa_seq_1 = seq_1
        aa_seq_2 = seq_2
        back_translated_aa_seq_1 = str(translate_to_dna_with_all_options(aa_seq_1))
        back_translated_aa_seq_2 = str(translate_to_dna_with_all_options(aa_seq_2))

    concatenated_sequence = process_sequences(aa_seq_1, aa_seq_2)
    if "Error" in concatenated_sequence:
        print(concatenated_sequence)
        return None

    print("The input amino acid sequences have been processed and concatenated:")
    print(concatenated_sequence)

    p = concatenated_sequence.split('*')
    formatted_aa_seq_1 = f"{p[0].strip()} * {p[1].strip()} *"
    formatted_aa_seq_2 = f"{p[1].strip()} * {p[0].strip()} *"

    # Build FC masks
    fc_mask_forward = None
    if fc_ranges_forward is not None:
        fc_mask_forward = _build_bool_mask_from_ranges(tok_len, fc_ranges_forward).flip(0)
    fc_mask_reverse = _build_bool_mask_from_ranges(tok_len, fc_ranges_reverse)

    if fixed_logits_spec_forward is not None:
        fixed_logits_spec_forward = shift_and_mirror_fixed_logits_positions_forward(fixed_logits_spec_forward, tok_len, model_numbers[0])
        print(fixed_logits_spec_forward)

    if fixed_logits_spec_reverse is not None:
        fixed_logits_spec_reverse = swap_fixed_logits_spec_classes_only_reverse(fixed_logits_spec_reverse, tok_len, model_numbers[0])
        print(fixed_logits_spec_reverse)


    all_results = []

    # --------------------------- FIRST PASS ---------------------------
    for model_num in model_numbers:
        model_path = model_data.loc[model_data['overlap_length'] == model_num, 'model_pth_location'].values[0]
        model = torch.load(model_path, map_location=device, weights_only=False).to(device)

        if inference_mode == "train":
            model.train()
        elif inference_mode == "eval":
            model.eval()
        else:
            raise ValueError("Invalid inference_mode. Choose 'train' or 'eval'.")

        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = attention_dropout_rate
            if isinstance(m, nn.Dropout):
                m.p = feedforward_dropout_rate
            if isinstance(m, nn.LayerNorm):
                m.eval()

        print("################################################################################################")
        print(f'Attempting to predict overlaps of length: {model_num}')
        print(f'This will be run {first_pass_iterations} time(s), then reversed.')
        print("################################################################################################")

        # Forward orientation
        fwd_handles = install_fc_token_dropout_and_constraints_hook(
            model,
            token_mask_1d=fc_mask_forward,
            p=fc_dropout_p,
            force_eval=apply_fc_dropout_in_eval,
            tok_len=tok_len,
            vocab_size_guess=vocab_size_guess,
            fixed_logits_spec=fixed_logits_spec_forward,
            fixed_value=fixed_value,
            blend_alpha=blend_alpha,
            debug=False
        )

        for i in range(first_pass_iterations):
            df = find_partially_matching_sequence(
                model,
                formatted_aa_seq_1,
                max_attempts_first_pass,
                first_pass_alignment_threshold_1,
                first_pass_alignment_threshold_2,
                first_pass_blosum_threshold_1,
                first_pass_blosum_threshold_2
            )
            if df is not None:
                all_results.append({
                    'model_number': model_num,
                    'pass': 'first',
                    'iteration': i + 1,
                    'modified_sequence_1': df['modified_sequence'].iloc[0],
                    'modified_sequence_2': df['modified_sequence'].iloc[1],
                    'overlap_length': df['overlap_length'].iloc[0],
                    'translated_aa_seq_1': df['translated_sequence'].iloc[0],
                    'translated_aa_seq_2': df['translated_sequence'].iloc[1],
                    'truncated_norm_align_score_1': round(df['truncated_norm_align_score_1'].iloc[0], 2),
                    'truncated_norm_align_score_2': round(df['truncated_norm_align_score_2'].iloc[0], 2),
                    'truncated_norm_align_avg_score': round(
                        (df['truncated_norm_align_score_1'].iloc[0] + df['truncated_norm_align_score_2'].iloc[0]) / 2, 2
                    ),
                    'blosum62_sim_score_1': round(df['blosum62_sim_score_1'].iloc[0], 2),
                    'blosum62_sim_score_2': round(df['blosum62_sim_score_2'].iloc[0], 2),
                    'blosum_sim_score_len_norm_1': round(df['blosum62_sim_score_1_len_norm'].iloc[0], 2),
                    'blosum_sim_score_len_norm_2': round(df['blosum62_sim_score_2_len_norm'].iloc[0], 2),
                    'blosum62_sum_score': round(df['blosum62_sim_score_1'].iloc[0] + df['blosum62_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_1': round(df['protsub_sim_score_1'].iloc[0], 2),
                    'protsub_sim_score_2': round(df['protsub_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_len_norm_1': round(df['protsub_sim_score_1_len_norm'].iloc[0], 2),
                    'protsub_sim_score_len_norm_2': round(df['protsub_sim_score_2_len_norm'].iloc[0], 2),
                    'protsub_sum_score': round(df['protsub_sim_score_1'].iloc[0] + df['protsub_sim_score_2'].iloc[0], 2)
                })

        for h in fwd_handles: h.remove()

        # Reverse orientation (NO MIRRORING if mask is None)
        rev_handles = install_fc_token_dropout_and_constraints_hook(
            model,
            token_mask_1d=fc_mask_reverse,
            p=fc_dropout_p,
            force_eval=apply_fc_dropout_in_eval,
            tok_len=tok_len,
            vocab_size_guess=vocab_size_guess,
            fixed_logits_spec=fixed_logits_spec_reverse,
            fixed_value=fixed_value,
            blend_alpha=blend_alpha,
            debug=False
        )

        for i in range(first_pass_iterations):
            df = find_partially_matching_sequence(
                model,
                formatted_aa_seq_2,
                max_attempts_first_pass,
                first_pass_alignment_threshold_2,
                first_pass_alignment_threshold_1,
                first_pass_blosum_threshold_2,
                first_pass_blosum_threshold_1
            )
            if df is not None:
                all_results.append({
                    'model_number': model_num,
                    'pass': 'first',
                    'iteration': i + 1,
                    'modified_sequence_1': df['modified_sequence'].iloc[1],
                    'modified_sequence_2': df['modified_sequence'].iloc[0],
                    'overlap_length': df['overlap_length'].iloc[0],
                    'translated_aa_seq_1': df['translated_sequence'].iloc[1],
                    'translated_aa_seq_2': df['translated_sequence'].iloc[0],
                    'truncated_norm_align_score_1': round(df['truncated_norm_align_score_2'].iloc[0], 2),
                    'truncated_norm_align_score_2': round(df['truncated_norm_align_score_1'].iloc[0], 2),
                    'truncated_norm_align_avg_score': round(
                        (df['truncated_norm_align_score_1'].iloc[0] + df['truncated_norm_align_score_2'].iloc[0]) / 2, 2
                    ),
                    'blosum62_sim_score_1': round(df['blosum62_sim_score_2'].iloc[0], 2),
                    'blosum62_sim_score_2': round(df['blosum62_sim_score_1'].iloc[0], 2),
                    'blosum_sim_score_len_norm_1': round(df['blosum62_sim_score_2_len_norm'].iloc[0], 2),
                    'blosum_sim_score_len_norm_2': round(df['blosum62_sim_score_1_len_norm'].iloc[0], 2),
                    'blosum62_sum_score': round(df['blosum62_sim_score_1'].iloc[0] + df['blosum62_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_1': round(df['protsub_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_2': round(df['protsub_sim_score_1'].iloc[0], 2),
                    'protsub_sim_score_len_norm_1': round(df['protsub_sim_score_2_len_norm'].iloc[0], 2),
                    'protsub_sim_score_len_norm_2': round(df['protsub_sim_score_1_len_norm'].iloc[0], 2),
                    'protsub_sum_score': round(df['protsub_sim_score_1'].iloc[0] + df['protsub_sim_score_2'].iloc[0], 2)
                })

        for h in rev_handles: h.remove()

    # --------------------------- FILTER ---------------------------
    results_df = pd.DataFrame(all_results)
    filtered = results_df[
        (results_df['truncated_norm_align_score_1'] >= first_pass_alignment_threshold_1) &
        (results_df['truncated_norm_align_score_2'] >= first_pass_alignment_threshold_2) &
        (results_df['overlap_length'] >= results_df['model_number'].apply(math.floor))
    ].drop_duplicates(subset=['model_number'])

    if filtered.empty:
        print("No models meet the alignment threshold.")
        return None

    final_results = []

    # --------------------------- SECOND PASS ---------------------------
    for _, row in filtered.iterrows():
        model_num = row['model_number']
        model_path = model_data.loc[model_data['overlap_length'] == model_num, 'model_pth_location'].values[0]
        model = torch.load(model_path, map_location=device, weights_only=False).to(device)

        if inference_mode == "train":
            model.train()
        else:
            model.eval()

        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                m.dropout = attention_dropout_rate
            if isinstance(m, nn.Dropout):
                m.p = feedforward_dropout_rate

        print("################################################################################################")
        print(f'Attempting to predict overlaps of length: {model_num}')
        print(f'This will be run {second_pass_iterations} time(s), then reversed.')
        print("################################################################################################")

        # Forward
        fwd_handles = install_fc_token_dropout_and_constraints_hook(
            model,
            token_mask_1d=fc_mask_forward,
            p=fc_dropout_p,
            force_eval=apply_fc_dropout_in_eval,
            tok_len=tok_len,
            vocab_size_guess=vocab_size_guess,
            fixed_logits_spec=fixed_logits_spec_forward,
            fixed_value=fixed_value,
            blend_alpha=blend_alpha,
            debug=False
        )

        for i in range(second_pass_iterations):
            df = find_partially_matching_sequence(
                model,
                formatted_aa_seq_1,
                max_attempts_second_pass,
                second_pass_alignment_threshold_1,
                second_pass_alignment_threshold_2,
                second_pass_blosum_threshold_1,
                second_pass_blosum_threshold_2
            )
            if df is not None:
                final_results.append({
                    'model_number': model_num,
                    'pass': 'second',
                    'iteration': i + 1,
                    'modified_sequence_1': df['modified_sequence'].iloc[0],
                    'modified_sequence_2': df['modified_sequence'].iloc[1],
                    'overlap_length': df['overlap_length'].iloc[0],
                    'translated_aa_seq_1': df['translated_sequence'].iloc[0],
                    'translated_aa_seq_2': df['translated_sequence'].iloc[1],
                    'truncated_norm_align_score_1': round(df['truncated_norm_align_score_1'].iloc[0], 2),
                    'truncated_norm_align_score_2': round(df['truncated_norm_align_score_2'].iloc[0], 2),
                    'truncated_norm_align_avg_score': round(
                        (df['truncated_norm_align_score_1'].iloc[0] + df['truncated_norm_align_score_2'].iloc[0]) / 2, 2
                    ),
                    'blosum62_sim_score_1': round(df['blosum62_sim_score_1'].iloc[0], 2),
                    'blosum62_sim_score_2': round(df['blosum62_sim_score_2'].iloc[0], 2),
                    'blosum_sim_score_len_norm_1': round(df['blosum62_sim_score_1_len_norm'].iloc[0], 2),
                    'blosum_sim_score_len_norm_2': round(df['blosum62_sim_score_2_len_norm'].iloc[0], 2),
                    'blosum62_sum_score': round(df['blosum62_sim_score_1'].iloc[0] + df['blosum62_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_1': round(df['protsub_sim_score_1'].iloc[0], 2),
                    'protsub_sim_score_2': round(df['protsub_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_len_norm_1': round(df['protsub_sim_score_1_len_norm'].iloc[0], 2),
                    'protsub_sim_score_len_norm_2': round(df['protsub_sim_score_2_len_norm'].iloc[0], 2),
                    'protsub_sum_score': round(df['protsub_sim_score_1'].iloc[0] + df['protsub_sim_score_2'].iloc[0], 2)
                })

        for h in fwd_handles: h.remove()

        # Reverse
        rev_handles = install_fc_token_dropout_and_constraints_hook(
            model,
            token_mask_1d=fc_mask_reverse,
            p=fc_dropout_p,
            force_eval=apply_fc_dropout_in_eval,
            tok_len=tok_len,
            vocab_size_guess=vocab_size_guess,
            fixed_logits_spec=fixed_logits_spec_reverse,
            fixed_value=fixed_value,
            blend_alpha=blend_alpha,
            debug=False
        )

        for i in range(second_pass_iterations):
            df = find_partially_matching_sequence(
                model,
                formatted_aa_seq_2,
                max_attempts_second_pass,
                second_pass_alignment_threshold_2,
                second_pass_alignment_threshold_1,
                second_pass_blosum_threshold_2,
                second_pass_blosum_threshold_1
            )
            if df is not None:
                final_results.append({
                    'model_number': model_num,
                    'pass': 'second',
                    'iteration': i + 1,
                    'modified_sequence_1': df['modified_sequence'].iloc[1],
                    'modified_sequence_2': df['modified_sequence'].iloc[0],
                    'overlap_length': df['overlap_length'].iloc[0],
                    'translated_aa_seq_1': df['translated_sequence'].iloc[1],
                    'translated_aa_seq_2': df['translated_sequence'].iloc[0],
                    'truncated_norm_align_score_1': round(df['truncated_norm_align_score_2'].iloc[0], 2),
                    'truncated_norm_align_score_2': round(df['truncated_norm_align_score_1'].iloc[0], 2),
                    'truncated_norm_align_avg_score': round(
                        (df['truncated_norm_align_score_1'].iloc[0] + df['truncated_norm_align_score_2'].iloc[0]) / 2, 2
                    ),
                    'blosum62_sim_score_1': round(df['blosum62_sim_score_2'].iloc[0], 2),
                    'blosum62_sim_score_2': round(df['blosum62_sim_score_1'].iloc[0], 2),
                    'blosum_sim_score_len_norm_1': round(df['blosum62_sim_score_2_len_norm'].iloc[0], 2),
                    'blosum_sim_score_len_norm_2': round(df['blosum62_sim_score_1_len_norm'].iloc[0], 2),
                    'blosum62_sum_score': round(df['blosum62_sim_score_1'].iloc[0] + df['blosum62_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_1': round(df['protsub_sim_score_2'].iloc[0], 2),
                    'protsub_sim_score_2': round(df['protsub_sim_score_1'].iloc[0], 2),
                    'protsub_sim_score_len_norm_1': round(df['protsub_sim_score_2_len_norm'].iloc[0], 2),
                    'protsub_sim_score_len_norm_2': round(df['protsub_sim_score_1_len_norm'].iloc[0], 2),
                    'protsub_sum_score': round(df['protsub_sim_score_1'].iloc[0] + df['protsub_sim_score_2'].iloc[0], 2)
                })

        for h in rev_handles: h.remove()

    final_results_df = pd.DataFrame(final_results + all_results)

    # post-processing
    if is_dna_sequence(seq_1) and is_dna_sequence(seq_2):
        final_results_df['integrated_seq_1'] = final_results_df.apply(
            lambda r: integrate_modified_sequence(seq_1, r['modified_sequence_1']), axis=1)
        final_results_df['integrated_seq_2'] = final_results_df.apply(
            lambda r: integrate_modified_sequence(seq_2, r['modified_sequence_2']), axis=1)
        final_results_df['translated_integrated_seq_1'] = final_results_df['integrated_seq_1'].apply(
            lambda s: str(Seq(s).translate()))
        final_results_df['translated_integrated_seq_2'] = final_results_df['integrated_seq_2'].apply(
            lambda s: str(Seq(s).translate()))
    else:
        final_results_df['integrated_seq_1'] = final_results_df.apply(
            lambda r: integrate_modified_sequence(back_translated_aa_seq_1, r['modified_sequence_1']), axis=1)
        final_results_df['integrated_seq_2'] = final_results_df.apply(
            lambda r: integrate_modified_sequence(back_translated_aa_seq_2, r['modified_sequence_2']), axis=1)
        final_results_df['translated_integrated_seq_1'] = final_results_df['integrated_seq_1'].apply(
            lambda s: str(Seq(s).translate()))
        final_results_df['translated_integrated_seq_2'] = final_results_df['integrated_seq_2'].apply(
            lambda s: str(Seq(s).translate()))

    print("\nFinal Results DataFrame:")
    print(final_results_df)
    return final_results_df


#################################################################################################
#################################################################################################

# Standard genetic code (NCBI #1) mapping AA -> possible codons
CODONS: Dict[str, List[str]] = {
    'A': ["GCT", "GCC", "GCA", "GCG"],
    'R': ["CGT", "CGC", "CGA", "CGG", "AGA", "AGG"],
    'N': ["AAT", "AAC"],
    'D': ["GAT", "GAC"],
    'C': ["TGT", "TGC"],
    'Q': ["CAA", "CAG"],
    'E': ["GAA", "GAG"],
    'G': ["GGT", "GGC", "GGA", "GGG"],
    'H': ["CAT", "CAC"],
    'I': ["ATT", "ATC", "ATA"],
    'L': ["TTA", "TTG", "CTT", "CTC", "CTA", "CTG"],
    'K': ["AAA", "AAG"],
    'M': ["ATG"],  # start Met
    'F': ["TTT", "TTC"],
    'P': ["CCT", "CCC", "CCA", "CCG"],
    'S': ["TCT", "TCC", "TCA", "TCG", "AGT", "AGC"],
    'T': ["ACT", "ACC", "ACA", "ACG"],
    'W': ["TGG"],
    'Y': ["TAT", "TAC"],
    'V': ["GTT", "GTC", "GTA", "GTG"],
    '*': ["TAA", "TAG", "TGA"],  # stop
    # Non-standard/ambiguous inputs handled below
}

# Very simple default codon usage (weights sum to 1 per amino acid). Replace with species-specific if desired.
DEFAULT_USAGE: Dict[str, Dict[str, float]] = {
    aa: {c: 1.0 / len(cs) for c in cs} for aa, cs in CODONS.items()
}
# Examples: bias a few toward common bacterial codons (user optional)
# DEFAULT_USAGE['L'] = {'CTG': 0.45, 'TTG': 0.15, 'TTA': 0.05, 'CTC': 0.1, 'CTA': 0.05, 'CTT': 0.2}
# DEFAULT_USAGE['A'] = {'GCT': 0.18, 'GCC': 0.4, 'GCA': 0.23, 'GCG': 0.19}
# DEFAULT_USAGE['G'] = {'GGT': 0.16, 'GGC': 0.41, 'GGA': 0.23, 'GGG': 0.20}
# DEFAULT_USAGE['R'] = {'CGT': 0.2, 'CGC': 0.35, 'CGA': 0.07, 'CGG': 0.07, 'AGA': 0.16, 'AGG': 0.15}

def _gc_fraction(dna: str) -> float:
    if not dna: return 0.0
    g = dna.count('G'); c = dna.count('C')
    return (g + c) / len(dna)

def _validate_usage(codon_usage: Dict[str, Dict[str, float]]) -> None:
    for aa, weights in codon_usage.items():
        if aa not in CODONS:  # allow only known keys
            continue
        allowed = set(CODONS[aa])
        if set(weights.keys()) - allowed:
            bad = set(weights.keys()) - allowed
            raise ValueError(f"Codon usage for {aa} contains invalid codons: {bad}")
        s = sum(weights.values())
        if s <= 0:
            raise ValueError(f"Codon usage weights for {aa} must sum to > 0")
        # Normalize in-place
        for k in weights:
            weights[k] /= s

def reverse_translate(
    aa_seq: str,
    *,
    strategy: str = "most_frequent",         # "most_frequent" | "random_weighted" | "gc_balanced"; user selected.
    codon_usage: Optional[Dict[str, Dict[str, float]]] = None,
    target_gc: Optional[float] = None,       # Used only if strategy == "gc_balanced"
    gc_tolerance: float = 0.01,              # How aggressively to steer GC
    add_start: bool = False,                 # Prepend ATG (overrides first AA only if you want to force ATG at start)
    end_with_stop: bool = False,             # Append a stop codon
    start_codon: str = "ATG",                # If add_start=True
    stop_codons_priority: Optional[List[str]] = None,  # Order for '*' or end stop; e.g., ["TAA","TGA","TAG"]
    rng: Optional[random.Random] = None,
    case: str = "upper"                      # "upper" or "lower"
) -> str:
    """
    Reverse-translate an amino-acid sequence to a DNA sequence.

    Ambiguity handling:
      - 'X': choose among all sense codons (excludes stops) via strategy/usage
      - 'B' (D/N): choose from D or N codons; 'Z' (E/Q): from E or Q codons; 'J' (L/I): from L or I codons
      - '*': choose a stop (or use stop_codons_priority)
      - 'U' treated as 'C' (selenocysteine not supported here)
    """
    if rng is None:
        rng = random.Random()

    seq = aa_seq.strip().upper()
    # Map a few ambiguous cases to allowed sets
    ambiguous_map: Dict[str, List[str]] = {
        'X': sum([v for k, v in CODONS.items() if k not in ('*')], []),
        'B': CODONS['D'] + CODONS['N'],
        'Z': CODONS['E'] + CODONS['Q'],
        'J': CODONS['L'] + CODONS['I'],
    }

    # Prepare codon usage
    usage = {aa: w.copy() for aa, w in (codon_usage or DEFAULT_USAGE).items()}
    # Ensure all keys exist
    for aa, cods in CODONS.items():
        usage.setdefault(aa, {c: 1.0 / len(cods) for c in cods})
    _validate_usage(usage)

    def pick_most_frequent(candidates: List[str], aa_for_usage: Optional[str]) -> str:
        if aa_for_usage and aa_for_usage in usage:
            weights = usage[aa_for_usage]
            # Filter to candidate set
            best = max(candidates, key=lambda c: weights.get(c, 0.0))
            return best
        # Fallback: just first candidate
        return candidates[0]

    def pick_weighted(candidates: List[str], aa_for_usage: Optional[str]) -> str:
        if aa_for_usage and aa_for_usage in usage:
            weights = [usage[aa_for_usage].get(c, 0.0) for c in candidates]
            s = sum(weights)
            if s <= 0:
                # Equal weights fallback
                idx = rng.randrange(len(candidates))
                return candidates[idx]
            # Normalize
            weights = [w / s for w in weights]
            r = rng.random()
            cum = 0.0
            for c, w in zip(candidates, weights):
                cum += w
                if r <= cum:
                    return c
            return candidates[-1]
        # Equal weight fallback
        return candidates[rng.randrange(len(candidates))]

    def pick_gc_balanced(candidates: List[str], aa_for_usage: Optional[str], built: str) -> str:
        # If no target, fallback to most_frequent
        if target_gc is None:
            return pick_most_frequent(candidates, aa_for_usage)
        # Score each candidate by how close the resulting GC would be to target;
        # break ties using usage weight (prefer more common codons).
        best_codon, best_score, best_usage = None, float("inf"), -1.0
        for c in candidates:
            gc_next = _gc_fraction(built + c)
            score = abs(gc_next - target_gc)
            u = usage.get(aa_for_usage, {}).get(c, 0.0) if aa_for_usage else 0.0
            # Prefer closer to target; on ties, higher usage
            if (score + gc_tolerance) < best_score or (abs(score - best_score) <= gc_tolerance and u > best_usage):
                best_codon, best_score, best_usage = c, score, u
        return best_codon or pick_most_frequent(candidates, aa_for_usage)

    def codon_for_symbol(sym: str, built: str) -> str:
        if sym == 'U':  # treat selenocysteine as cysteine here (no SEC machinery modeled)
            sym = 'C'
        if sym == '*':
            candidates = stop_codons_priority or CODONS['*']
            # For stops, just take priority order
            return candidates[0]
        if sym in CODONS:
            candidates = CODONS[sym]
            aa_key = sym
        elif sym in ambiguous_map:
            candidates = ambiguous_map[sym]
            aa_key = None  # usage not meaningful across mixed AAs
        else:
            raise ValueError(f"Unsupported amino-acid symbol: '{sym}'")

        if strategy == "most_frequent":
            return pick_most_frequent(candidates, aa_key)
        elif strategy == "random_weighted":
            return pick_weighted(candidates, aa_key)
        elif strategy == "gc_balanced":
            return pick_gc_balanced(candidates, aa_key, built)
        else:
            raise ValueError("strategy must be one of {'most_frequent','random_weighted','gc_balanced'}")

    dna_chunks: List[str] = []

    # Optional forced start codon
    if add_start:
        dna_chunks.append(start_codon.upper())

    for i, aa in enumerate(seq):
        # If add_start and the first AA is M, you may want to skip picking another codon for position 1.
        if i == 0 and add_start and aa == 'M':
            continue
        dna_chunks.append(codon_for_symbol(aa, "".join(dna_chunks)))

    if end_with_stop and (not seq or seq[-1] != '*'):
        stop_choice = (stop_codons_priority or CODONS['*'])[0]
        dna_chunks.append(stop_choice)

    dna = "".join(dna_chunks)
    if case == "lower":
        dna = dna.lower()
    return dna


def get_bracket_positions_logits(sequence_with_brackets: str) -> List[Tuple[int, str]]:
    """
    Extract (start_index, bracketed_aa) pairs where index 0 is the amino acid
    that is 104 residues in from the C-terminus. Indices increase toward the N-terminus.

    Keeps any bracket that overlaps the region within min_allowed AA from the C-terminal,
    and trims the N-terminal side if part of the bracket is outside that zone.
    """
    seq2_len = sum(1 for ch in sequence_with_brackets if ch not in "[]")
    anchor_i = seq2_len - 104
    if anchor_i < 0:
        raise ValueError(f"Sequence too short ({seq2_len} aa) to be 104 in from tail.")

    def map_index(i: int) -> int:
        return i - anchor_i

    min_allowed = (model_numbers[0] // 3) - 2  # AA from C-terminal

    bracketed_amino_acids: List[Tuple[int, str]] = []
    pos_with_brackets = 0
    i_seq2 = 0  # AA index in unbracketed seq

    while pos_with_brackets < len(sequence_with_brackets):
        ch = sequence_with_brackets[pos_with_brackets]

        if ch == '[':
            start_index_unbr = i_seq2
            pos_with_brackets += 1
            content = []

            while (pos_with_brackets < len(sequence_with_brackets) and
                   sequence_with_brackets[pos_with_brackets] != ']'):
                content.append(sequence_with_brackets[pos_with_brackets])
                pos_with_brackets += 1
                i_seq2 += 1

            if pos_with_brackets >= len(sequence_with_brackets):
                raise ValueError("Unmatched '[' in sequence.")

            pos_with_brackets += 1
            end_index_unbr = i_seq2 - 1

            # Distances from C-terminal for each residue in this bracket
            distances = [seq2_len - (start_index_unbr + k) for k in range(len(content))]

            # Find first AA within allowed zone
            keep_start_idx = None
            for idx, dist in enumerate(distances):
                if dist <= min_allowed:
                    keep_start_idx = idx
                    break

            if keep_start_idx is not None:
                trimmed_content = ''.join(content[keep_start_idx:])
                start_index_mapped = map_index(start_index_unbr + keep_start_idx)
                bracketed_amino_acids.append((start_index_mapped, trimmed_content))

        elif ch == ']':
            raise ValueError("Unmatched ']' in sequence.")
        else:
            pos_with_brackets += 1
            i_seq2 += 1

    return bracketed_amino_acids


NUC_TO_NUM = {"A": 1, "T": 2, "G": 3, "C": 4}

## Testing modified approach
def bracket_aas_to_logits_spec(
    sequence_with_brackets: str,
    *,
    strategy: str = "random_weighted",
    codon_usage: Optional[Dict[str, Dict[str, float]]] = None,
    target_gc: Optional[float] = None,
    rng: Optional[random.Random] = None
) -> Dict[int, int]:
    """
    Convert bracketed AA segments to a fixed_logits_spec dict where keys are
    nucleotide positions (0-based) and values are numeric codes:
        T=1, A=2, G=3, C=4
    """
    positions: List[Tuple[int, str]] = get_bracket_positions_logits(sequence_with_brackets)
    fixed_logits_spec: Dict[int, int] = {}

    for start_idx, aa_seg in positions:
        dna_seq = reverse_translate(
            aa_seg,
            strategy=strategy,
            codon_usage=codon_usage,
            target_gc=target_gc,
            rng=rng,
            case="upper"
        )
        # Map AA start to nucleotide start
        nuc_start = start_idx * 3
        for offset, base in enumerate(dna_seq):
            fixed_logits_spec[nuc_start + offset] = NUC_TO_NUM[base]

    return fixed_logits_spec



COMPLEMENT = str.maketrans('ATGC', 'TACG')

def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    return seq.translate(COMPLEMENT)[::-1]



def bracket_aas_to_logits_spec_flipped(
    sequence_with_brackets: str,
    *,
    strategy: str = "random_weighted",
    codon_usage: Optional[Dict[str, Dict[str, float]]] = None,
    target_gc: Optional[float] = None,
    rng: Optional[random.Random] = None,
    tok_len: int = 315,
    frame_shift: Optional[int] = None   # 0..2; if None we derive from model_index
) -> Dict[int, int]:
    """
    Map bracketed AA segments from the SECOND sequence to fixed logits on the FORWARD token axis,
    using reverse complement AND mirrored positions.

    For a segment starting at nt_start with length L, the flipped start is:
        start_flipped = tok_len - (nt_start + L)
    Then we write rc(dna) left-to-right starting at start_flipped.
    """
    positions: List[Tuple[int, str]] = get_bracket_positions_logits(sequence_with_brackets)
    fixed_logits_spec: Dict[int, int] = {}

    # Optional small shift (0..2) to keep codon frames aligned between strands.
    if frame_shift is None:
        if model_numbers[0] is not None:

            frame_shift = ((tok_len) - model_numbers[0])
        else:
            frame_shift = 0

    for aa_start, aa_seg in positions:
        # 1) Translate, 2) RC
        dna = reverse_translate(
            aa_seg,
            strategy=strategy,
            codon_usage=codon_usage,
            target_gc=target_gc,
            rng=rng,
            case="upper",
        )
        rc = reverse_complement(dna)
        Lnt = len(rc)

        # 3) Original nt start (left-indexed AA positions)
        nt_start = aa_start * 3

        # 4) Mirror about the right edge + optional frame shift; NO modulo wrap
        start_flipped = tok_len - (nt_start + Lnt) + frame_shift

        # 5) Bounds check (fail fast instead of silently wrapping)
        if start_flipped < 0 or start_flipped + Lnt > tok_len:
            raise ValueError(
                f"Flipped segment out of bounds: start={start_flipped}, len={Lnt}, tok_len={tok_len} "
                f"(aa_start={aa_start}, aa_len={len(aa_seg)}, frame_shift={frame_shift})"
            )

        for o, base in enumerate(rc):
            fixed_logits_spec[start_flipped + o] = NUC_TO_NUM[base]

    return fixed_logits_spec


# ================================================================================================
# --------------------------- BEGIN: batch runner for aa pair optimization -----------------------
# ================================================================================================


"""
Batch runner for amino-acid pair table (CSV/XLSX) with ESM integration and
configurable secondary-structure (SS) evaluation behavior.

This script was designed to support overlapping gene design experiments, where
two amino acid sequences (potentially overlapping in different reading frames)
are optimized and scored on multiple metrics:
    - Secondary structure (SS) similarity
    - Amino acid alignment identity / substitution scores
    - Contact map similarity (via ESM-2 SSIM, Structural Similarity Index)

The workflow allows for "deferred" computation of heavy structural predictions,
so users can control runtime costs. Caching is heavily used to prevent
recomputing SS or embeddings for the same sequences across rows.

"""

# ===== Global flag controlling INITIAL vs SUBSEQUENT behaviour =====
# Global flag to track if any prior window had success
had_successful_prediction = False


# ESM (facebookresearch/esm)
try:
    import esm
    import traceback
    import os
    _ESM_AVAILABLE = True
except Exception as _e:
    esm = None
    _ESM_AVAILABLE = False
    print(f"[WARN] ESM not available: {_e}")

# Guard against accidental regression weight loading
if hasattr(esm.pretrained, "load_regression_model_and_alphabet_local"):
    esm.pretrained.load_regression_model_and_alphabet_local = lambda *a, **k: (
        (_ for _ in ()).throw(RuntimeError("Regression loader disabled")),
        None,
    )

overlap_length_selected = [311] # This is the default aoverlap length; define new value later to override.

model_numbers = overlap_length_selected

# ==============================
# Config
# ==============================
# Secondary-structure control:
DO_SS = True                    # If True: run SS for all candidates (original heavy behaviour)
EVAL_SS_ON_EARLY = True          # If DO_SS==False: evaluate SS for the early-stop candidate (if found)
EVAL_SS_AT_ATTEMPT_END = True    # If DO_SS==False: evaluate SS for the chosen final candidate at attempt end

# ESM thresholds (0..100)
ESM_EARLY_THRESHOLD = 92.0
ESM_SUCCESS_THRESHOLD = 92.0

# Whether to reset heavy caches per-row (default False to avoid recomputing across many rows)
RESET_CACHES_PER_ROW = True

# Composite weight parameters (must sum to 1.0 ideally)
WEIGHTS = {
    "ss":   0.15,
    "sub":  0.15,
    "aln":  0.10,
    "esm":  0.60,
}

# ==============================
# IO / Utility functions
# ==============================

def windows_to_wsl_path(p: str) -> str:
    p = p.strip().strip('"').strip("'")
    if re.match(r'^[A-Za-z]:\\', p):
        drive = p[0].lower()
        rest = p[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"
    return p.replace('\\', '/')

def load_pairs_table(path: str) -> pd.DataFrame:

    """
    Load an input table containing pair rows.
    Accepts .xlsx/.xls and .csv. Verifies required columns exist:
        - aa_seq_1
        - aa_seq_2
        - aa_seq_1_brackets ###If no brackets are defind, this can be the same as aa_seq_1
        - aa_seq_2_brackets ###If no brackets are defind, this can be the same as aa_seq_2

    Returns:
        pandas.DataFrame loaded from file.
    Raises:
        ValueError for unsupported extensions.
        KeyError if required columns missing.
    """

    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext in (".csv",):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    required = {"aa_seq_1", "aa_seq_2", "aa_seq_1_brackets", "aa_seq_2_brackets"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")
    return df

# ───────────────────────────────────────────────
# 1. Global caches and metrics (persist across rows)
# ───────────────────────────────────────────────
SSCACHE: Dict[str, Any] = {}          # sequence -> secondary structure prediction (raw)
METRICS = {
    "ss_total_requests": 0,
    "ss_unique_requests": 0,
    "ss_cache_hits": 0,
    "pairs_requested": 0,
    "pairs_unique": 0,
}
all_pairs_records: List[Dict[str, Any]] = []

# Helper to update records with deferred SS results
def update_all_pairs_records_with_ss(target_seq1: str, target_seq2: str,
                                     attempt: int, window: int,
                                     s1_pct: float, s2_pct: float,
                                     ss_pred_a: Optional[str] = None,
                                     ss_pred_b: Optional[str] = None,
                                     avg_pct: Optional[float] = None) -> int:
    """
    Update all_pairs_records in-place for records matching:
      - translated_integrated_seq_1 == target_seq1 (trimmed, trailing '*' removed)
      - translated_integrated_seq_2 == target_seq2
      - attempt == attempt
      - window == window

    Writes:
      ss_score_1, ss_score_2, ss_score_avg, ss_pred_1, ss_pred_2

    Returns:
      number of records updated (int)

    Note:
      Matches on normalized strings to tolerate trailing '*' and whitespace.
    """
    updated = 0
    t1 = str(target_seq1).strip().rstrip("*")
    t2 = str(target_seq2).strip().rstrip("*")
    for rec in all_pairs_records:
        if (str(rec.get("translated_integrated_seq_1", "")).strip().rstrip("*") == t1
                and str(rec.get("translated_integrated_seq_2", "")).strip().rstrip("*") == t2
                and int(rec.get("attempt", -1)) == int(attempt)
                and int(rec.get("window", -999)) == int(window)):
            rec["ss_score_1"] = float(s1_pct)
            rec["ss_score_2"] = float(s2_pct)
            rec["ss_score_avg"] = float(avg_pct) if avg_pct is not None else (float(s1_pct) + float(s2_pct)) / 2.0
            rec["ss_pred_1"] = ss_pred_a
            rec["ss_pred_2"] = ss_pred_b
            updated += 1
    return updated

# ───────────────────────────────────────────────
# 2. Secondary structure caching (user must supply batch_structure_prediction_wrapper)
# ───────────────────────────────────────────────

def ss_predict_cached(seq_list: Iterable[str]) -> List[Any]:
    """
    Deduplicate and cache SS predictions for a batch of sequences.

    Workflow:
      - Count requests in METRICS
      - Find which sequences are cache misses
      - Call batch_structure_prediction_wrapper(misses, output_dir) for misses
      - Store returned preds in SSCACHE
      - Return predictions in input order (pulling from cache)

    Requirements:
      - The function batch_structure_prediction_wrapper(misses, output_dir) must exist
        in the user's environment and return predictions in the same order as misses.
      - `output_dir` must be defined in the calling context (this module uses it as a global)

    Returns:
      List of predictions aligned to seq_list order. Predictions may be None for failures.
    """
    global METRICS, SSCACHE
    seq_list = list(seq_list)
    METRICS["ss_total_requests"] += len(seq_list)
    misses = []
    for s in seq_list:
        if s in SSCACHE:
            METRICS["ss_cache_hits"] += 1
        elif s not in misses:
            # Ensure we only request each distinct miss once
            misses.append(s)
    if misses:
        METRICS["ss_unique_requests"] += len(misses)
        preds = batch_structure_prediction_wrapper(misses)
        for s, p in zip(misses, preds):
            SSCACHE[s] = p
    # Return cached preds in the same order as the requested list
    return [SSCACHE[s] for s in seq_list]

# ───────────────────────────────────────────────
# 3. ESM integration (embeddings + contact maps) and helpers
# ───────────────────────────────────────────────
ESM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ESM_MODEL_NAME = "esm2_t12_35M_UR50D"  # pick smaller if memory limited, e.g. "esm2_t12_35M_UR50D"
ESM_LAYER = 12
ESM_BATCH_SIZE = 8

# Performance knobs
USE_AUTOCast_FP16_IF_CUDA = True
USE_MODEL_FP16 = False

# Global ESM caches (persist across rows by default)
ESM_EMB_CACHE: Dict[str, np.ndarray] = {}   # sequence -> embedding (mean-pooled)
ESM_CONT_CACHE: Dict[str, np.ndarray] = {}  # sequence -> contact map (L_res x L_res)

def _load_esm_model():
    """
    Lazily loads the selected ESM model and returns (model, alphabet, batch_converter)
    - If esm.pretrained has a direct attribute for the chosen name, use it, otherwise call known loader.
    - Places model on ESM_DEVICE and sets eval() mode.
    - Optionally converts model to half precision if USE_MODEL_FP16 is True and a CUDA device is present.

    Raises:
      RuntimeError if ESM not available.

    Returns:
      (model, alphabet, batch_converter)
    """
    if not _ESM_AVAILABLE:
        raise RuntimeError("ESM not installed; pip install fair-esm")
    loader = getattr(esm.pretrained, ESM_MODEL_NAME, None)
    if loader is None:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    else:
        model, alphabet = loader()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(ESM_DEVICE)
    if USE_MODEL_FP16 and ESM_DEVICE.type == "cuda":
        model = model.half()
    return model, alphabet, batch_converter

def _chunked(lst, n):
    """
    Simple generator to yield chunks (sublists) of size up to `n`.
    Keeps memory use bounded for batched processing.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _meanpool_token_representations(token_reps: torch.Tensor, tokens: torch.Tensor, padding_idx: int) -> torch.Tensor:
    """
    Mean-pool token representations while ignoring padding and special tokens.
    - token_reps: (B, L, C) tensor of token-level representations
    - tokens: (B, L) long tensor with token indices (to detect padding)
    - padding_idx: alphabet.padding_idx value

    Behavior:
      - Excludes first and last positions (often BOS/EOS in ESM)
      - Excludes any padding positions
      - Returns (B, C) pooled tensor as sums / (counts) per sequence
      - clamps count to at least 1 to avoid divide-by-zero
    """
    B, L, C = token_reps.shape
    nonpad = (tokens != padding_idx)
    idx = torch.arange(L, device=tokens.device)
    valid_pos = nonpad & (idx[None, :] > 0) & (idx[None, :] < (L - 1))
    sums = (token_reps * valid_pos.unsqueeze(-1)).sum(dim=1)
    counts = valid_pos.sum(dim=1).clamp(min=1)
    return sums / counts.unsqueeze(-1)

def _contacts_to_numpy(contact_batch: torch.Tensor, lens: torch.Tensor) -> list:
    """
    Convert model contacts output into numpy 2D contact matrices for each sequence.

    The model returns contact tensors with padding included; this extracts
    the residue-residue slice corresponding to the real sequence length (excluding special tokens).
    Returns:
      list of numpy arrays (L_res x L_res), one per sequence in batch
    """
    outs = []
    for cm, le in zip(contact_batch, lens):
        L = int(le.item()) - 2
        L = max(L, 1)
        cm_res = cm[1:1+L, 1:1+L]
        outs.append(cm_res.detach().float().cpu().numpy())
    return outs

def esm_features_cached(seq_list: Iterable[str], model_alphabet_converter=None):
    """
    Ensure ESM_EMB_CACHE and ESM_CONT_CACHE entries exist for the sequences in seq_list.

    Behavior:
      - Normalizes sequences (strip spaces, trailing '*', skip empty)
      - Finds cache misses
      - Loads model via _load_esm_model() unless model_alphabet_converter passed
      - Batches inference, extracts:
          - mean-pooled token representations (embedding)
          - contact maps (2D)
      - Stores results in ESM_EMB_CACHE and ESM_CONT_CACHE keyed by sequence string.

    Note:
      - This function intentionally populates caches and returns None.
      - On failure for a chunk, it will raise (caller may catch).
    """
    global ESM_EMB_CACHE, ESM_CONT_CACHE
    seq_list = [s.strip().replace(" ", "").rstrip("*") for s in seq_list if isinstance(s, str) and len(s)]
    misses = [s for s in seq_list if (s not in ESM_EMB_CACHE) or (s not in ESM_CONT_CACHE)]
    if not misses:
        return

    if model_alphabet_converter is None:
        mac = _load_esm_model()
    else:
        mac = model_alphabet_converter
    model, alphabet, batch_converter = mac

    with torch.no_grad():
        for chunk in _chunked(misses, ESM_BATCH_SIZE):
            data = [(f"seq_{i}", s) for i, s in enumerate(chunk)]
            labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(ESM_DEVICE)
            lens = (batch_tokens != alphabet.padding_idx).sum(1)

            if USE_AUTOCast_FP16_IF_CUDA and ESM_DEVICE.type == "cuda":
                # modern API — avoids the FutureWarning and is forward compatible
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=True)
            else:
                out = model(batch_tokens, repr_layers=[ESM_LAYER], return_contacts=True)

            reps = out["representations"][ESM_LAYER]
            contacts = out["contacts"]

            pooled = _meanpool_token_representations(reps, batch_tokens, alphabet.padding_idx)
            pooled = pooled.detach().to(torch.float32).cpu().numpy()
            c_maps = _contacts_to_numpy(contacts, lens)

            for s, emb, cm in zip(chunk, pooled, c_maps):
                ESM_EMB_CACHE[s] = emb
                ESM_CONT_CACHE[s] = cm

def _crop_to_min(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop two square matrices to the same minimal square size using their first dimension.
    As implemented, given that both sequences will be the same length, this will not have much/any impact.
    If not the same size, useful because SSIM and other matrix comparisons require shapes to align.
    """
    m = min(a.shape[0], b.shape[0])
    return a[:m, :m], b[:m, :m]

def clamp01(x, eps=1e-9):
    return float(np.clip(x, eps, 1.0 - eps))

def transform_ssim(x, method="power", gamma=1, temp=1.0, batch=None):
    """
    Map a single SSIM x (0..1) to a transformed 0..1 value.
    If x is nan, return nan.
    Methods: "power","logit","linear_minmax","quantile","quantile_norm","tanh_z"
    - batch required for quantile/linear_minmax/tanh_z/quantile_norm
    This is optional
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    x = clamp01(x)

    if method == "power":
        return x ** gamma

    if method == "logit":
        # amplify differences near 0/1 via logit/temperature
        logit = np.log(x / (1.0 - x))
        scaled = logit / float(max(1e-9, temp))
        return float(expit(scaled))

    if method == "linear_minmax":
        if batch is None:
            raise ValueError("batch required for linear_minmax")
        arr = np.asarray([v for v in batch if not (np.isnan(v))], dtype=float)
        if arr.size == 0:
            return x  # nothing to scale
        mn, mx = arr.min(), arr.max()
        if mx <= mn:
            return 0.0 if x <= mn else 1.0
        return float(np.clip((x - mn) / (mx - mn), 0.0, 1.0))

    if method == "quantile":
        if batch is None:
            raise ValueError("batch required for quantile")
        arr = np.asarray(batch, dtype=float)
        pct = float((arr < x).sum()) / max(1, len(arr))
        return pct

    if method == "quantile_norm":
        if batch is None:
            raise ValueError("batch required for quantile_norm")
        arr = np.asarray(batch, dtype=float)
        pct = float((arr < x).sum()) / max(1, len(arr))
        z = norm.ppf(np.clip(pct, 1e-6, 1-1e-6))
        return float(0.5 * (np.tanh(z / (temp if temp > 0 else 1.0)) + 1.0))

    if method == "tanh_z":
        if batch is None:
            return x ** gamma
        arr = np.asarray(batch, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 2:
            return x ** gamma
        mu, sigma = arr.mean(), arr.std() if arr.std() > 0 else 1.0
        z = (x - mu) / (sigma * (temp if temp>0 else 1.0))
        return float(0.5 * (np.tanh(z) + 1.0))

    return x

def esm_pair_metrics(orig_seq: str, cand_seq: str) -> Dict[str, float]:
    cm_o = ESM_CONT_CACHE.get(orig_seq, None)
    cm_c = ESM_CONT_CACHE.get(cand_seq, None)

    if cm_o is None or cm_c is None:
        print(f"[DEBUG] Missing contact map(s): "
              f"{'orig_seq missing' if cm_o is None else ''} "
              f"{'cand_seq missing' if cm_c is None else ''}")
        return {"cmap_ssim_100": 50.0, "esm_score_100": 50.0}

    try:
        # Ensure float32 for consistency
        cm_o = np.asarray(cm_o, dtype=np.float32)
        cm_c = np.asarray(cm_c, dtype=np.float32)

        # Fixed range: contact maps assumed normalized to [0,1]
        ssim_raw, _ = ssim(cm_o, cm_c, full=True, data_range=1.0)
        ssim_val = transform_ssim(ssim_raw, method="power", gamma=1.0)

        return {
            "cmap_ssim_100": round(ssim_val * 100.0, 2),
            "esm_score_100": round(ssim_val * 100.0, 2),
        }

    except Exception as e:
        print(f"[ERROR] SSIM computation failed for pair "
              f"({orig_seq[:8]}..., {cand_seq[:8]}...): {e}")
        return {"cmap_ssim_100": 50.0, "esm_score_100": 50.0}

# -------------------------
# Helper: compute SS for a chosen pair (uses cached ss_predict_cached)
# -------------------------
def _compute_ss_for_pair_and_scores(ori_1_predicted_structure, ori_2_predicted_structure,
                                    seq_a: str, seq_b: str, model_id: int):
    """
    For a chosen candidate pair (seq_a, seq_b):
      - Ensure SS predictions via ss_predict_cached for seq_a and seq_b.
      - Call compare_sequences(...) which performs the SS comparison of candidate vs original predicted structures.
      - Unpack and return the percentage scores and raw SS predictions.

    Returns:
      (s1_pct, s2_pct, avg_pct, ss_pred_a, ss_pred_b)

    Note:
      - compare_sequences must exist in the environment and is expected to return a tuple whose
        positions are unpacked here. This wrapper isolates the caching and common unpack logic.
    """
    seqs = [seq_a, seq_b]
    preds = ss_predict_cached(seqs)
    ss_a, ss_b = preds[0], preds[1]
    *_, s1_pct, _, s2_pct, avg_pct, _ = compare_sequences(
        ori_1_predicted_structure, ori_2_predicted_structure,
        ss_a, ss_b, model_id
    )
    return float(s1_pct), float(s2_pct), float(avg_pct), ss_a, ss_b

# ======================================================================================
# 4. Core single-pair optimization wrapper (with ESM scoring + optional deferred SS eval)
# ======================================================================================
def optimize_pair_and_save(seq1: str,
                           seq2: str,
                           seq1_bracket:str,
                           seq2_bracket:str,
                           row_index_1based: int,
                           weights: Dict[str, float],
                           *,
                           first_pass_iterations: int = 1,
                           second_pass_iterations: int = 75) -> str:

    """
    Main pipeline for processing a single input row (pair of AA sequences).

    Responsibilities:
      - Prepare sequences and optionally compute original SS predictions.
      - Warm-up/load ESM and populate caches for originals.
      - Iterate over windows and run the inference engine (user-provided run_inference_for_models)
      - Score candidates using a weighted composite (SS, alignment, substitution, ESM)
      - Optionally perform deferred SS computation (if DO_SS==False)
      - Persist all candidate rows to an Excel file named by date/model/row index

    Inputs:
      - seq1, seq2: raw AA sequences from the table (may contain whitespace or trailing '*')
      - seq1_bracket, seq2_bracket: bracket annotations used to set fixed_logits specs
      - row_index_1based: integer for output filename hygiene

    Returns:
      - path to saved .xlsx result output for this row
    """
    # --- Local/default params (kept similar to your original script) ---

    # --- Declare globals before any assignment ---
    global all_pairs_records, output_dir, METRICS

    had_successful_prediction = False

    # --- Reset per-row records so nothing leaks across rows ---
    all_pairs_records = []

    SUB_MATRIX = "blosum62"
    LOGIT_DROPOUT_RATE = 0.25
    TOK_LEN = 315

    first_pass_alignment_threshold_1 = 0.34
    first_pass_alignment_threshold_2 = 0.34
    second_pass_alignment_threshold_1 = 0.34
    second_pass_alignment_threshold_2 = 0.34
    first_pass_blosum_threshold_1 = 0
    first_pass_blosum_threshold_2 = 0
    second_pass_blosum_threshold_1 = 0
    second_pass_blosum_threshold_2 = 0

    max_attempts_first_pass = 325
    max_attempts_second_pass = 200
    inference_mode = "train"
    set_seed = None

    MAX_RETRIES_PER_WINDOW = 3
    APPLY_LOGIT_DROPOUT_IN_EVAL = True

    WINDOW_AA, STRIDE_AA, MODEL_AA = 10, 8, 105
    num_w = math.ceil(MODEL_AA / STRIDE_AA)

    SUCCESS_SS_THRESHOLD = 100.0
    EARLY_SS_THRESHOLD = 100.0
    MAX_TOTAL_RETRIES = 2

    FEEDFORWARD_DROPOUT_INITIAL = 0.1
    ATTENTION_DROPOUT_INITIAL = 0.1
    FEEDFORWARD_DROPOUT_SUBSEQ = 0.0
    ATTENTION_DROPOUT_SUBSEQ = 0.0

    output_dir = working_directory

    # use originals (no brackets here)
    sequence_1_original = seq1.strip().replace(" ", "").strip("*")
    sequence_2_original = seq2.strip().replace(" ", "").strip("*")
    sequence_1_aa_brackets = seq1_bracket.strip().replace(" ", "").strip("*")
    sequence_2_aa_brackets = seq2_bracket.strip().replace(" ", "").strip("*")

    def _preview(s: str, n: int = 120) -> str:
        if not s:
            return "<EMPTY>"
        return s if len(s) <= n else f"{s[:n]}... (len={len(s)})"

    # Print diagnostics to debug delimiter/format problems quickly
    print("[INPUT DIAG] sequence_1_original preview:", _preview(sequence_1_original))
    print("           length:", len(sequence_1_original), "  '*' count:", sequence_1_original.count("*"))
    print("[INPUT DIAG] sequence_2_original preview:", _preview(sequence_2_original))
    print("           length:", len(sequence_2_original), "  '*' count:", sequence_2_original.count("*"))

    print("[INPUT DIAG] sequence_1_aa_brackets preview:", _preview(sequence_1_aa_brackets))
    print("           length:", len(sequence_1_aa_brackets), "  '*' count:", sequence_1_aa_brackets.count("*"))
    print("[INPUT DIAG] sequence_2_aa_brackets preview:", _preview(sequence_2_aa_brackets))
    print("           length:", len(sequence_2_aa_brackets), "  '*' count:", sequence_2_aa_brackets.count("*"))

    # Optionally reset caches per-row (default False)
    global SSCACHE, ESM_EMB_CACHE, ESM_CONT_CACHE
    if RESET_CACHES_PER_ROW:
        SSCACHE = {}
        ESM_EMB_CACHE = {}
        ESM_CONT_CACHE = {}

    print("\n[INFO] Processing sequences:")
    # Precompute originals and optionally SS for originals (only if DO_SS True or if deferred eval will need them later)
    concatenated_sequence_original = process_sequences(sequence_1_original, sequence_2_original)
    print("\n[INFO] Concatenated original sequence (nucleotides):")
    print(concatenated_sequence_original)

    ori_seq_1, ori_seq_2 = split_sequence(concatenated_sequence_original)
    print("\n[INFO] Split sequences (AA):")
    print("  Seq1:", ori_seq_1)
    print("  Seq2:", ori_seq_2)

    ori_1_predicted_structure = None
    ori_2_predicted_structure = None
    if DO_SS:
        ori_1_predicted_structure = predict_secondary_structure(ori_seq_1)
        ori_2_predicted_structure = predict_secondary_structure(ori_seq_2)
        print("\n[INFO] Secondary structure predictions (orig computed):")
        print("  Seq1 secondary structure:", ori_1_predicted_structure)
        print("  Seq2 secondary structure:", ori_2_predicted_structure)
    else:
        print("\n[INFO] Secondary-structure scoring deferred (DO_SS is False).")

    # ESM warm-up for originals (cache them)
    if _ESM_AVAILABLE:
        try:
            mac = _load_esm_model()
            esm_features_cached([ori_seq_1, ori_seq_2], model_alphabet_converter=mac)
        except Exception as _e:
            mac = None
            print(f"[WARN] ESM loading/feature extraction failed; continuing without ESM. Error: {_e}")
    else:
        mac = None
        print("[WARN] ESM not available; ESM-derived scores will be neutral.")


    def _has_real_seq(seq_list):
        """
        Return True if seq_list contains at least one non-empty, non-whitespace, non-'*' sequence.
        Filters out None, NaN, empty strings, whitespace-only strings, and strings consisting only of '*'s.
        """
        for s in seq_list:
            if s is None:
                continue
            # guard against pandas NaN (float)
            if isinstance(s, float) and np.isnan(s):
                continue
            if not isinstance(s, str):
                s = str(s)
            cleaned = s.strip().strip("*").strip()
            if cleaned:
                return True
        return False

    # Note: safe_run_inference uses the run_inference_for_models function
    def safe_run_inference(fwd_range, cur_seq1_input, cur_seq2_input):
        """
        Wraps run_inference_for_models with:
          - retry loop up to MAX_RETRIES_PER_WINDOW
          - dynamic dropout selection depending on whether previous windows succeeded
          - combined fixed logits spec built from bracket annotations

        Returns:
          DataFrame from run_inference_for_models on success, or None if all retries failed.

        Note:
          - run_inference_for_models is expected to be defined in the environment and accept
            the long list of parameters passed here.
        """

        if not had_successful_prediction:
            print("[INFO] Using INITIAL dropout/alignment thresholds (this may be slower than subsequent windows).", file=sys.__stdout__)
            ff_drop = FEEDFORWARD_DROPOUT_INITIAL
            att_drop = ATTENTION_DROPOUT_INITIAL
            phase = "INITIAL"
        else:
            print("[INFO] Prior window success detected; using SUBSEQUENT dropout/alignment thresholds for all remaining runs.", file=sys.__stdout__)
            ff_drop = FEEDFORWARD_DROPOUT_SUBSEQ
            att_drop = ATTENTION_DROPOUT_SUBSEQ
            phase = "SUBSEQUENT"

            print(f"[Dropout State: {phase}] Feedforward={ff_drop}, Attention={att_drop}, Sub Matrix={SUB_MATRIX}", file=sys.__stdout__)


        # Build combined fixed logits spec once per attempt (keeps same semantics)
        fixed_logits_spec_seq1 = bracket_aas_to_logits_spec(sequence_1_aa_brackets, rng=None)
        fixed_logits_spec_seq2 = bracket_aas_to_logits_spec_flipped(sequence_2_aa_brackets, rng=None)
        combined_logits = {**fixed_logits_spec_seq1, **fixed_logits_spec_seq2}

        for attempt in range(1, MAX_RETRIES_PER_WINDOW + 1):
            buf = io.StringIO()
            try:
                with redirect_stdout(buf):
                    df = run_inference_for_models(
                        model_numbers,
                        cur_seq1_input, cur_seq2_input,
                        max_attempts_first_pass, max_attempts_second_pass,
                        first_pass_alignment_threshold_1, first_pass_alignment_threshold_2,
                        second_pass_alignment_threshold_1, second_pass_alignment_threshold_2,
                        first_pass_blosum_threshold_1, first_pass_blosum_threshold_2,
                        second_pass_blosum_threshold_1, second_pass_blosum_threshold_2,
                        first_pass_iterations, second_pass_iterations,
                        inference_mode,
                        ff_drop, att_drop,
                        set_seed,
                        fc_dropout_p=LOGIT_DROPOUT_RATE,
                        fc_ranges_forward=fwd_range,
                        fc_ranges_reverse=fwd_range,
                        tok_len=TOK_LEN,
                        apply_fc_dropout_in_eval=APPLY_LOGIT_DROPOUT_IN_EVAL,
                        fixed_logits_spec_forward=combined_logits,
                        fixed_logits_spec_reverse=combined_logits,
                        fixed_value=12.0,
                        blend_alpha=None
                    )
                return df
            except Exception as e:
                # Print error and optionally the last lines of the buffered stdout for debugging
                debug_out = buf.getvalue()
                print(f"    [{attempt}/{MAX_RETRIES_PER_WINDOW}] Inference attempt failed: {e}")
                if debug_out:
                    # Show a short preview of the last buffered line to help debugging
                    last_line = debug_out.strip().splitlines()[-1] if debug_out.strip() else ""
                    print(f"    [debug stdout last line] {last_line}")
                # If we have more attempts, continue to next loop iteration and retry
        # Exhausted attempts
        print(f"    All {MAX_RETRIES_PER_WINDOW} inference attempts failed for window {fwd_range}; returning None.")
        return None

    # Main optimization loop across retry attempts (restarts entire window sweep up to MAX_TOTAL_RETRIES)
    retry_count = 0
    success = False

    start_seq_aa_1 = sequence_1_original
    start_seq_aa_2 = sequence_2_original

    best_overall_scores = (0.0, 0.0)
    best_overall_seq1 = start_seq_aa_1
    best_overall_seq2 = start_seq_aa_2
    best_overall_summaries = []

    while retry_count < MAX_TOTAL_RETRIES:
        retry_count += 1
        print(f"\n=== Optimization attempt {retry_count}/{MAX_TOTAL_RETRIES} ===")
        summaries = []
        early_halt = False

        for w in range(num_w):
            print(f"Entering window {w+1}/{num_w}")
            start_aa, end_aa = w * STRIDE_AA, min((w * STRIDE_AA) + WINDOW_AA, MODEL_AA)
            fwd_range = [(start_aa * 3, end_aa * 3 - 1)]

            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    df = safe_run_inference(fwd_range, start_seq_aa_1, start_seq_aa_2)
            except Exception as e:
                print(f"    Inference failed for window {w+1}: {e}")
                df = None

            if df is None or df.empty:
                print("    All retries failed or no candidates; skipping this window.")
                continue

            seqs1_all = df["translated_aa_seq_1"].astype(str).tolist()
            seqs2_all = df["translated_aa_seq_2"].astype(str).tolist()

            # Deduplicate pairs
            unique_pairs, seen_pair_keys = [], set()
            for s1, s2 in zip(seqs1_all, seqs2_all):
                if (s1, s2) not in seen_pair_keys:
                    seen_pair_keys.add((s1, s2))
                    unique_pairs.append((s1, s2))

            METRICS["pairs_requested"] += len(seqs1_all)
            METRICS["pairs_unique"] += len(unique_pairs)

            seqs1 = [p[0] for p in unique_pairs]
            seqs2 = [p[1] for p in unique_pairs]

            if DO_SS:
                combined = seqs1 + seqs2
                combined_preds = ss_predict_cached(combined)

                # Split back into two aligned lists
                ss1_list = combined_preds[:len(seqs1)]
                ss2_list = combined_preds[len(seqs1):]
            else:
                ss1_list = [None] * len(seqs1)
                ss2_list = [None] * len(seqs2)

            # ESM batch compute features for candidates (contacts + embeddings)
            if _ESM_AVAILABLE and mac is not None:
                try:
                    esm_features_cached(set([ori_seq_1, ori_seq_2] + seqs1 + seqs2), model_alphabet_converter=mac)
                except Exception as _e:
                    print(f"[WARN] ESM feature extraction failed this window: {_e}")

            # If we got *any* valid sequence predictions, flip the flag and adjust thresholds for subsequent runs
            if not had_successful_prediction and _has_real_seq(seqs1_all) and _has_real_seq(seqs2_all):

                had_successful_prediction = True
                print("[STATE] Initial sequence prediction detected — future windows will use SUBSEQUENT alignment thresholds.")

                # Adjust thresholds for subsequent passes (example values; adjust as required)
                first_pass_alignment_threshold_1  = 0.80
                first_pass_alignment_threshold_2  = 0.80
                second_pass_alignment_threshold_1 = 0.80
                second_pass_alignment_threshold_2 = 0.80

                print(
                    f"FirstPass=({first_pass_alignment_threshold_1}, {first_pass_alignment_threshold_2}), "
                    f"SecondPass=({second_pass_alignment_threshold_1}, {second_pass_alignment_threshold_2})"
                )

            scores = []
            best_idx = None
            best_combined = -1e9
            early_idx = None

            # Weight schedule – pull from config / passed-in dict
            ss_weight   = weights["ss"] if DO_SS else 0.0
            sub_weight  = weights["sub"]
            align_weight = weights["aln"]
            esm_weight  = weights["esm"]

            matrix_key = SUB_MATRIX.strip().lower()
            if matrix_key in ("blosum", "blosum62", "blosum_62"):
                sim_func = calculate_blosum62_similarity
            elif matrix_key in ("protsub", "prot-sub", "prot_sub"):
                sim_func = calculate_protsub_similarity
            else:
                raise ValueError(f"Unsupported SUB_MATRIX: {SUB_MATRIX}")

            model_id = model_numbers[0]
            for i, (s1, s2, ss1_pred, ss2_pred) in enumerate(zip(seqs1, seqs2, ss1_list, ss2_list)):
                # Secondary-structure comparison (only when DO_SS True)
                if DO_SS and (ss1_pred is not None) and (ss2_pred is not None):
                    *_, s1_pct, _, s2_pct, avg_pct, _ = compare_sequences(
                        ori_1_predicted_structure, ori_2_predicted_structure,
                        ss1_pred, ss2_pred, model_id
                    )
                else:
                    s1_pct = 0.0
                    s2_pct = 0.0
                    avg_pct = 0.0

                s1_clean = s1.rstrip("*")
                s2_clean = s2.rstrip("*")
                align1_pct = (align_sequences_identity(ori_seq_1, s1_clean) / len(s1_clean)) * 100 if s1_clean else 0.0
                align2_pct = (align_sequences_identity(ori_seq_2, s2_clean) / len(s2_clean)) * 100 if s2_clean else 0.0

                try:
                    den1 = float(sim_func(ori_seq_1, ori_seq_1))
                except Exception:
                    den1 = 0.0
                try:
                    den2 = float(sim_func(ori_seq_2, ori_seq_2))
                except Exception:
                    den2 = 0.0

                sub1_pct = round((float(sim_func(ori_seq_1, s1_clean)) / den1) * 100, 2) if (s1_clean and den1 != 0.0) else 0.0
                sub2_pct = round((float(sim_func(ori_seq_2, s2_clean)) / den2) * 100, 2) if (s2_clean and den2 != 0.0) else 0.0

                # ESM metrics
                if _ESM_AVAILABLE and mac is not None:
                    try:
                        m1 = esm_pair_metrics(ori_seq_1, s1_clean)
                        m2 = esm_pair_metrics(ori_seq_2, s2_clean)
                        esm1 = m1["esm_score_100"]
                        esm2 = m2["esm_score_100"]
                        esm_avg = round((esm1 + esm2) / 2.0, 2)
                    except Exception:
                        esm1 = esm2 = esm_avg = 50.0
                else:
                    esm1 = esm2 = esm_avg = 50.0

                combined_score = (
                    (avg_pct * ss_weight) +
                    (((align1_pct + align2_pct) / 2.0) * align_weight) +
                    (((sub1_pct + sub2_pct) / 2.0) * sub_weight) +
                    (esm_avg * esm_weight) #+

                )

                scores.append((
                    s1_pct, s2_pct, avg_pct,
                    align1_pct, align2_pct,
                    sub1_pct, sub2_pct,
                    esm1, esm2, esm_avg,
                    combined_score
                ))

                if combined_score > best_combined:
                    best_combined = combined_score
                    best_idx = i

                # Early-stop logic: prefer SS if DO_SS True; otherwise ESM threshold
                if DO_SS:
                    if s1_pct >= EARLY_SS_THRESHOLD and s2_pct >= EARLY_SS_THRESHOLD and early_idx is None:
                        early_idx = i
                        print(f"      Early-stop candidate at i={i}: SS1={s1_pct:.2f}%, SS2={s2_pct:.2f}%")
                        break
                else:
                    if (esm1 >= ESM_EARLY_THRESHOLD) and (esm2 >= ESM_EARLY_THRESHOLD) and early_idx is None:
                        early_idx = i
                        print(f"      Early-stop candidate at i={i}: ESM1={esm1:.2f}, ESM2={esm2:.2f}")
                        break

            # Build df_unique and attach metrics
            processed_n = len(scores)

            df_unique = (
                df.drop_duplicates(subset=["translated_aa_seq_1", "translated_aa_seq_2"])
                .reset_index(drop=True)
            )

            # Trim to processed_n to stay in sync with scores length
            df_unique = df_unique.iloc[:processed_n].copy()

            if scores:
                (df_unique["ss_score_1"], df_unique["ss_score_2"], df_unique["ss_score_avg"],
                 df_unique["align1"], df_unique["align2"],
                 df_unique["sub1"], df_unique["sub2"],
                 df_unique["esm1"], df_unique["esm2"], df_unique["esm_avg"],
                 df_unique["combined_score"]) = zip(*scores)

            # Map integrated sequences if present
            if "translated_integrated_seq_1" in df.columns and "translated_integrated_seq_2" in df.columns:
                pair_to_integrated = {}
                for r in df.itertuples(index=False):
                    key = (r.translated_aa_seq_1, r.translated_aa_seq_2)
                    if key not in pair_to_integrated:
                        pair_to_integrated[key] = (r.translated_integrated_seq_1, r.translated_integrated_seq_2)
                df_unique["translated_integrated_seq_1"] = [
                    pair_to_integrated.get((a, b), (a, b))[0].rstrip("*")
                    for a, b in zip(df_unique["translated_aa_seq_1"], df_unique["translated_aa_seq_2"])
                ]
                df_unique["translated_integrated_seq_2"] = [
                    pair_to_integrated.get((a, b), (a, b))[1].rstrip("*")
                    for a, b in zip(df_unique["translated_aa_seq_1"], df_unique["translated_aa_seq_2"])
                ]
            else:
                df_unique["translated_integrated_seq_1"] = [s.rstrip("*") for s in df_unique["translated_aa_seq_1"]]
                df_unique["translated_integrated_seq_2"] = [s.rstrip("*") for s in df_unique["translated_aa_seq_2"]]

            # Cleaned originals (no '*')
            seq1_clean = sequence_1_original.replace("*", "")
            seq2_clean = sequence_2_original.replace("*", "")

            # Take terminal 104 aa if available
            seq1_terminal_104 = seq1_clean[-104:] if len(seq1_clean) > 104 else seq1_clean
            seq2_terminal_104 = seq2_clean[-104:] if len(seq2_clean) > 104 else seq2_clean

            # Append records with attempt/window metadata (these may have SS placeholders if DO_SS==False)
            for i, row in enumerate(df_unique.itertuples(index=False)):
                all_pairs_records.append({
                    "aa_seq_1_original": sequence_1_original,
                    "aa_seq_2_original": sequence_2_original,

                    # NEW: terminal subsequences
                    "aa_seq_1_terminal_104": seq1_terminal_104,
                    "aa_seq_2_terminal_104": seq2_terminal_104,

                    "translated_aa_seq_1": row.translated_aa_seq_1,
                    "translated_aa_seq_2": row.translated_aa_seq_2,
                    "modified_sequence_1": row.modified_sequence_1,
                    "modified_sequence_2": row.modified_sequence_2,
                    "translated_integrated_seq_1": row.translated_integrated_seq_1,
                    "translated_integrated_seq_2": row.translated_integrated_seq_2,
                    "integrated_seq_1": row.integrated_seq_1,
                    "integrated_seq_2": row.integrated_seq_2,

                    # scores
                    "ss_score_1": float(getattr(row, "ss_score_1", 0.0)),
                    "ss_score_2": float(getattr(row, "ss_score_2", 0.0)),
                    "ss_score_avg": float(getattr(row, "ss_score_avg", 0.0)),
                    "ss_pred_1": ss1_list[i] if DO_SS else None,
                    "ss_pred_2": ss2_list[i] if DO_SS else None,
                    "align1": float(getattr(row, "align1", 0.0)),
                    "align2": float(getattr(row, "align2", 0.0)),
                    "sub1": float(getattr(row, "sub1", 0.0)),
                    "sub2": float(getattr(row, "sub2", 0.0)),
                    "esm1": float(getattr(row, "esm1", np.nan)),
                    "esm2": float(getattr(row, "esm2", np.nan)),
                    "esm_avg": float(getattr(row, "esm_avg", np.nan)),
                    "combined_score": float(getattr(row, "combined_score", 0.0)),
                    "attempt": retry_count,
                    "window": w,
                    "attempt_window": f"{retry_count}-{w+1}"
                })

            # --- Ensure parent is considered without recomputing heavy metrics ---
            # config (near top of script)
            INCLUDE_PARENT_AS_CANDIDATE = False   # Set False to NOT include parent as a candidate

            # --- Ensure parent is considered without recomputing heavy metrics (optional) ---
            parent_key1 = str(start_seq_aa_1).rstrip("*").strip()
            parent_key2 = str(start_seq_aa_2).rstrip("*").strip()

            parent_mask = (
                df_unique["translated_integrated_seq_1"].astype(str).str.rstrip("*").str.strip() == parent_key1
            ) & (
                df_unique["translated_integrated_seq_2"].astype(str).str.rstrip("*").str.strip() == parent_key2
            )

            # Try to find cached parent record (cheap in-memory lookup)
            parent_rec = None
            for rec in all_pairs_records:
                if (str(rec.get("translated_integrated_seq_1", "")).strip().rstrip("*") == parent_key1
                        and str(rec.get("translated_integrated_seq_2", "")).strip().rstrip("*") == parent_key2):
                    parent_rec = rec
                    break

            if INCLUDE_PARENT_AS_CANDIDATE:
                if parent_mask.any():
                    # Parent already present in df_unique: update its recorded heavy metrics from cache (if available)
                    if parent_rec is not None:
                        # Update relevant columns for rows matching parent_key
                        mask = parent_mask
                        # Ensure columns exist
                        for col in ["ss_score_1","ss_score_2","ss_score_avg","ss_pred_1","ss_pred_2",
                                    "align1","align2","sub1","sub2",
                                    "esm1","esm2","esm_avg",
                                    "combined_score"]:
                            if col not in df_unique.columns:
                                df_unique[col] = 0.0 if col != "ss_pred_1" and col != "ss_pred_2" else None
                        # Assign from cached parent_rec (use get with defaults)
                        df_unique.loc[mask, "ss_score_1"] = parent_rec.get("ss_score_1", df_unique.loc[mask, "ss_score_1"])
                        df_unique.loc[mask, "ss_score_2"] = parent_rec.get("ss_score_2", df_unique.loc[mask, "ss_score_2"])
                        df_unique.loc[mask, "ss_score_avg"] = parent_rec.get("ss_score_avg", df_unique.loc[mask, "ss_score_avg"])
                        df_unique.loc[mask, "ss_pred_1"] = parent_rec.get("ss_pred_1", df_unique.loc[mask, "ss_pred_1"])
                        df_unique.loc[mask, "ss_pred_2"] = parent_rec.get("ss_pred_2", df_unique.loc[mask, "ss_pred_2"])
                        df_unique.loc[mask, "align1"] = parent_rec.get("align1", df_unique.loc[mask, "align1"])
                        df_unique.loc[mask, "align2"] = parent_rec.get("align2", df_unique.loc[mask, "align2"])
                        df_unique.loc[mask, "sub1"] = parent_rec.get("sub1", df_unique.loc[mask, "sub1"])
                        df_unique.loc[mask, "sub2"] = parent_rec.get("sub2", df_unique.loc[mask, "sub2"])
                        df_unique.loc[mask, "esm1"] = parent_rec.get("esm1", df_unique.loc[mask, "esm1"])
                        df_unique.loc[mask, "esm2"] = parent_rec.get("esm2", df_unique.loc[mask, "esm2"])
                        df_unique.loc[mask, "esm_avg"] = parent_rec.get("esm_avg", df_unique.loc[mask, "esm_avg"])
                        # update combined_score only if cached parent has a numeric combined_score
                        try:
                            parent_comb = float(parent_rec.get("combined_score", np.nan))
                            if not np.isnan(parent_comb):
                                df_unique.loc[mask, "combined_score"] = parent_comb
                        except Exception:
                            pass

                else:
                    # Parent not present in df_unique: append cached parent row (if available)
                    if parent_rec is not None:
                        append_row = {
                            "translated_aa_seq_1": parent_rec.get("translated_aa_seq_1", parent_key1),
                            "translated_aa_seq_2": parent_rec.get("translated_aa_seq_2", parent_key2),
                            "translated_integrated_seq_1": parent_rec.get("translated_integrated_seq_1", parent_key1),
                            "translated_integrated_seq_2": parent_rec.get("translated_integrated_seq_2", parent_key2),
                            "ss_score_1": parent_rec.get("ss_score_1", 0.0),
                            "ss_score_2": parent_rec.get("ss_score_2", 0.0),
                            "ss_score_avg": parent_rec.get("ss_score_avg", 0.0),
                            "align1": parent_rec.get("align1", 0.0),
                            "align2": parent_rec.get("align2", 0.0),
                            "sub1": parent_rec.get("sub1", 0.0),
                            "sub2": parent_rec.get("sub2", 0.0),
                            "esm1": parent_rec.get("esm1", 50.0),
                            "esm2": parent_rec.get("esm2", 50.0),
                            "esm_avg": parent_rec.get("esm_avg", 50.0),
                            "combined_score": parent_rec.get("combined_score", 0.0),
                            "ss_pred_1": parent_rec.get("ss_pred_1", None),
                            "ss_pred_2": parent_rec.get("ss_pred_2", None),
                            "attempt": parent_rec.get("attempt", retry_count),
                            "window": parent_rec.get("window", w),
                            "attempt_window": parent_rec.get("attempt_window", f"{retry_count}-{w+1}")
                        }
                        df_unique = pd.concat([df_unique, pd.DataFrame([append_row])], ignore_index=True, sort=False)

            # Make combined_score numeric & NaN-safe then pick best candidate (same logic as before)
            if "combined_score" not in df_unique.columns:
                df_unique["combined_score"] = 0.0
            df_unique["combined_score"] = pd.to_numeric(df_unique["combined_score"], errors="coerce").fillna(-1e12)

            # Pick chosen_row: early_idx prioritized
            if early_idx is not None:
                chosen_row = df_unique.iloc[early_idx].copy()
                chosen_index = early_idx
                reason = "early stop"
            else:
                chosen_index = int(df_unique["combined_score"].idxmax())
                chosen_row = df_unique.loc[chosen_index].copy()
                reason = "best combined_score"

            # If DO_SS==False but EVAL_SS_ON_EARLY True and we had early candidate -> compute SS for that candidate now
            if (not DO_SS) and (early_idx is not None) and EVAL_SS_ON_EARLY:
                try:
                    # ensure original SS available
                    if ori_1_predicted_structure is None or ori_2_predicted_structure is None:
                        ori_1_predicted_structure = predict_secondary_structure(ori_seq_1)
                        ori_2_predicted_structure = predict_secondary_structure(ori_seq_2)

                    s1_pct, s2_pct, avg_pct, ss_a, ss_b = _compute_ss_for_pair_and_scores(
                        ori_1_predicted_structure, ori_2_predicted_structure,
                        str(chosen_row["translated_integrated_seq_1"]),
                        str(chosen_row["translated_integrated_seq_2"]),
                        model_id
                    )
                    chosen_row["ss_score_1"] = s1_pct
                    chosen_row["ss_score_2"] = s2_pct
                    chosen_row["ss_score_avg"] = avg_pct
                    # Recompute combined_score

                    chosen_row["combined_score"] = (
                        (avg_pct * ss_weight) +
                        (((chosen_row["align1"] + chosen_row["align2"]) / 2.0) * align_weight) +
                        (((chosen_row["sub1"] + chosen_row["sub2"]) / 2.0) * sub_weight) +
                        (chosen_row["esm_avg"] * esm_weight)
                    )

                    # Update df_unique so saved bookkeeping reflects computed SS
                    if chosen_index is not None:
                        df_unique.at[chosen_index, "ss_score_1"] = chosen_row["ss_score_1"]
                        df_unique.at[chosen_index, "ss_score_2"] = chosen_row["ss_score_2"]
                        df_unique.at[chosen_index, "ss_score_avg"] = chosen_row["ss_score_avg"]
                        df_unique.at[chosen_index, "combined_score"] = chosen_row["combined_score"]
                    else:
                        # Parent selected but not present as an index in this df_unique — update any matching integrated pair rows
                        key1 = str(chosen_row["translated_integrated_seq_1"]).rstrip("*").strip()
                        key2 = str(chosen_row["translated_integrated_seq_2"]).rstrip("*").strip()
                        mask = (
                            df_unique["translated_integrated_seq_1"].astype(str).str.rstrip("*").str.strip() == key1
                        ) & (
                            df_unique["translated_integrated_seq_2"].astype(str).str.rstrip("*").str.strip() == key2
                        )
                        if mask.any():
                            df_unique.loc[mask, "ss_score_1"] = chosen_row["ss_score_1"]
                            df_unique.loc[mask, "ss_score_2"] = chosen_row["ss_score_2"]
                            df_unique.loc[mask, "ss_score_avg"] = chosen_row["ss_score_avg"]
                            df_unique.loc[mask, "combined_score"] = chosen_row["combined_score"]
                    print(f"    [EVAL SS ON EARLY] Computed SS for early candidate: SS1={s1_pct:.3f}, SS2={s2_pct:.3f}, new combined={chosen_row['combined_score']:.3f}")

                    # Update the in-memory records so the saved Excel includes these SS values
                    updated = update_all_pairs_records_with_ss(
                        target_seq1=str(chosen_row["translated_integrated_seq_1"]),
                        target_seq2=str(chosen_row["translated_integrated_seq_2"]),
                        attempt=retry_count,
                        window=w,
                        s1_pct=s1_pct,
                        s2_pct=s2_pct,
                        ss_pred_a=ss_a,
                        ss_pred_b=ss_b,
                        avg_pct=avg_pct
                    )
                    print(f"    [EVAL SS ON EARLY] Updated {updated} entry(ies) in all_pairs_records with deferred SS.")
                except Exception as _e:
                    print(f"[WARN] SS evaluation for early candidate failed: {_e}")

            print(
                f"    Selected ({reason}): combined={chosen_row['combined_score']:.3f}, "
                f"Align1={chosen_row['align1']:.2f}%, Align2={chosen_row['align2']:.2f}%, "
                f"ESM_avg={chosen_row['esm_avg']:.2f}%"
            )

            start_seq_aa_1 = str(chosen_row["translated_integrated_seq_1"])
            start_seq_aa_2 = str(chosen_row["translated_integrated_seq_2"])

            summaries.append(dict(
                window = w,
                AA_range = f"{start_aa}-{end_aa}",
                SS_score1 = round(float(chosen_row.get("ss_score_1", 0.0)), 3),
                SS_score2 = round(float(chosen_row.get("ss_score_2", 0.0)), 3),
                SS_avg = round(float(chosen_row.get("ss_score_avg", 0.0)), 3),
                Align1 = round(float(chosen_row["align1"]), 2),
                Align2 = round(float(chosen_row["align2"]), 2),
                Sub1 = round(float(chosen_row["sub1"]), 2),
                Sub2 = round(float(chosen_row["sub2"]), 2),
                ESM1 = round(float(chosen_row["esm1"]), 2),
                ESM2 = round(float(chosen_row["esm2"]), 2),
                ESM_avg = round(float(chosen_row["esm_avg"]), 2),
                Combined = round(float(chosen_row["combined_score"]), 3),
            ))
            print(f"  Summary so far: {summaries[-1]}")

            if early_idx is not None:
                print(f"Early success detected; halting this attempt.")
                early_halt = True
                break

        # End windows

        if early_halt:
            success = True
            # If we halted early, optionally evaluate SS for the final chosen pair at attempt end
            if (not DO_SS) and EVAL_SS_AT_ATTEMPT_END:
                try:
                    final_a = start_seq_aa_1
                    final_b = start_seq_aa_2
                    if ori_1_predicted_structure is None or ori_2_predicted_structure is None:
                        ori_1_predicted_structure = predict_secondary_structure(ori_seq_1)
                        ori_2_predicted_structure = predict_secondary_structure(ori_seq_2)
                    f_s1_pct, f_s2_pct, f_avg_pct, f_ss_a, f_ss_b = _compute_ss_for_pair_and_scores(
                        ori_1_predicted_structure, ori_2_predicted_structure,
                        final_a, final_b, model_id
                    )
                    print(f"[EVAL SS AT ATTEMPT END] Final pair SS: seq1={f_s1_pct:.3f}, seq2={f_s2_pct:.3f}, avg={f_avg_pct:.3f}")

                    # Update summaries' last window entry if available
                    if summaries:
                        summaries[-1]["SS_score1"] = round(f_s1_pct, 3)
                        summaries[-1]["SS_score2"] = round(f_s2_pct, 3)
                        summaries[-1]["SS_avg"] = round(f_avg_pct, 3)

                    # Update stored records for the final chosen pair - use last summary window if available
                    final_window_index = summaries[-1]["window"] if summaries else (w if 'w' in locals() else 0)
                    updated = update_all_pairs_records_with_ss(
                        target_seq1=final_a,
                        target_seq2=final_b,
                        attempt=retry_count,
                        window=final_window_index,
                        s1_pct=f_s1_pct,
                        s2_pct=f_s2_pct,
                        ss_pred_a=f_ss_a,
                        ss_pred_b=f_ss_b,
                        avg_pct=f_avg_pct
                    )
                    print(f"[EVAL SS AT ATTEMPT END] Updated {updated} entry(ies) in all_pairs_records with final SS.")
                except Exception as _e:
                    print(f"[WARN] Deferred SS computation at attempt end failed: {_e}")
            break

        if not summaries:
            print("\nNo summaries generated this attempt.")
            continue

        final_summary = summaries[-1]
        final_ss1, final_ss2 = final_summary["SS_score1"], final_summary["SS_score2"]
        final_esm1, final_esm2 = final_summary["ESM1"], final_summary["ESM2"]
        print(f"\nFinal metrics this attempt: SS (seq1,seq2)=({final_ss1},{final_ss2}), ESM (seq1,seq2)=({final_esm1},{final_esm2})")

        # If DO_SS False but we want SS at attempt end, compute SS for final chosen pair now
        if (not DO_SS) and EVAL_SS_AT_ATTEMPT_END:
            try:
                final_a = start_seq_aa_1
                final_b = start_seq_aa_2
                if ori_1_predicted_structure is None or ori_2_predicted_structure is None:
                    ori_1_predicted_structure = predict_secondary_structure(ori_seq_1)
                    ori_2_predicted_structure = predict_secondary_structure(ori_seq_2)
                f_s1_pct, f_s2_pct, f_avg_pct, f_ss_a, f_ss_b = _compute_ss_for_pair_and_scores(
                    ori_1_predicted_structure, ori_2_predicted_structure,
                    final_a, final_b, model_id
                )
                summaries[-1]["SS_score1"] = round(f_s1_pct, 3)
                summaries[-1]["SS_score2"] = round(f_s2_pct, 3)
                summaries[-1]["SS_avg"] = round(f_avg_pct, 3)
                print(f"[EVAL SS AT_ATTEMPT_END] Computed SS for attempt-final pair: SS1={f_s1_pct:.3f}, SS2={f_s2_pct:.3f}")

                # Update the stored records for the final chosen pair:
                final_window_index = final_summary["window"]
                updated = update_all_pairs_records_with_ss(
                    target_seq1=final_a,
                    target_seq2=final_b,
                    attempt=retry_count,
                    window=final_window_index,
                    s1_pct=f_s1_pct,
                    s2_pct=f_s2_pct,
                    ss_pred_a=f_ss_a,
                    ss_pred_b=f_ss_b,
                    avg_pct=f_avg_pct
                )
                print(f"[EVAL SS AT_ATTEMPT_END] Updated {updated} entry(ies) in all_pairs_records with attempt-final SS.")
            except Exception as _e:
                print(f"[WARN] Deferred SS computation at attempt end failed: {_e}")

        # Update best overall using appropriate metric
        if DO_SS:
            score_for_compare = summaries[-1]["SS_score1"] + summaries[-1]["SS_score2"]
            if score_for_compare > sum(best_overall_scores):
                best_overall_scores = (summaries[-1]["SS_score1"], summaries[-1]["SS_score2"])
                best_overall_seq1, best_overall_seq2 = start_seq_aa_1, start_seq_aa_2
                best_overall_summaries = summaries.copy()
        else:
            if (final_esm1 + final_esm2) > sum(best_overall_scores):
                best_overall_scores = (final_esm1, final_esm2)
                best_overall_seq1, best_overall_seq2 = start_seq_aa_1, start_seq_aa_2
                best_overall_summaries = summaries.copy()

        # Stopping decision
        if DO_SS:
            if final_ss1 >= SUCCESS_SS_THRESHOLD and final_ss2 >= SUCCESS_SS_THRESHOLD:
                print(f"Success: Both sequences exceed {SUCCESS_SS_THRESHOLD}% SS. Halting.")
                success = True
                break
            else:
                print(f"Secondary structure below {SUCCESS_SS_THRESHOLD}% — retrying…")
        else:
            if (final_esm1 >= ESM_SUCCESS_THRESHOLD) and (final_esm2 >= ESM_SUCCESS_THRESHOLD):
                print(f"Success: Both sequences exceed {ESM_SUCCESS_THRESHOLD} ESM SSIM. Halting.")
                success = True
                break
            else:
                print(f"ESM SSIM below {ESM_SUCCESS_THRESHOLD} for final candidate — retrying…")

    # End attempts loop

    # ---------------------------------------------------------
    # Deferred SS: evaluate top-N by ESM and write back to records
    # (Only when DO_SS is False and we deferred SS during run)
    # ---------------------------------------------------------
    def _evaluate_top_n_by_esm_and_compute_ss(top_n: int = 20):
        """
        When SS computations were deferred (DO_SS==False), this function:
          - selects the top-N candidates by esm_avg (descending),
          - computes SS predictions for those candidates,
          - writes the SS metrics back into matching all_pairs_records entries.

        Matching strategy:
          - Prefer exact match by translated_integrated_seq pair + attempt + window.
          - If exact match not found, performs a best-effort fallback to any matching seq pair.

        Returns:
          number of records updated
        """
        nonlocal ori_1_predicted_structure, ori_2_predicted_structure, ori_seq_1, ori_seq_2, model_id
        global all_pairs_records
        if DO_SS:
            print("[INFO] DO_SS True -> no deferred top-N SS computation required.")
            return 0

        if not all_pairs_records:
            print("[INFO] No pairs recorded; skipping deferred top-N SS computation.")
            return 0

        df_all = pd.DataFrame(all_pairs_records)

        # Require esm_avg present and numeric
        if "esm_avg" not in df_all.columns:
            print("[INFO] esm_avg column missing; skipping deferred top-N SS computation.")
            return 0

        df_candidates = df_all[~df_all["esm_avg"].isna()].copy()
        if df_candidates.empty:
            print("[INFO] No candidates with esm_avg available; skipping deferred top-N SS computation.")
            return 0

        # Sort descending and take top_n unique (by integrated seq pair + attempt + window)
        df_candidates.sort_values("esm_avg", ascending=False, inplace=True)
        # Keep top_n rows (preserve attempt/window so updates target exact entries)
        df_top = df_candidates.head(top_n)

        # Ensure original SS preds exist (we need these for compare_sequences)
        try:
            if ori_1_predicted_structure is None:
                ori_1_predicted_structure = predict_secondary_structure(ori_seq_1)
            if ori_2_predicted_structure is None:
                ori_2_predicted_structure = predict_secondary_structure(ori_seq_2)
        except Exception as _e:
            print(f"[WARN] Could not compute original SS predictions required for deferred scoring: {_e}")
            # Continue: _compute_ss_for_pair_and_scores will attempt per-sequence calls via ss_predict_cached.

        updated_count = 0
        attempted_count = 0
        for r in df_top.itertuples(index=False):
            attempted_count += 1
            try:
                # Prefer translated_integrated_seq if present (consistent with earlier updates)
                seq1_t = getattr(r, "translated_integrated_seq_1", None) or getattr(r, "translated_aa_seq_1", None)
                seq2_t = getattr(r, "translated_integrated_seq_2", None) or getattr(r, "translated_aa_seq_2", None)
                if seq1_t is None or seq2_t is None:
                    print(f"  [SKIP] Missing sequence in top candidate row: {r}")
                    continue

                attempt_val = int(getattr(r, "attempt", -1))
                window_val = int(getattr(r, "window", -999))

                # Compute SS and similarity scores for this pair
                try:
                    s1_pct, s2_pct, avg_pct, ss_a, ss_b = _compute_ss_for_pair_and_scores(
                        ori_1_predicted_structure, ori_2_predicted_structure,
                        str(seq1_t), str(seq2_t), model_id
                    )
                except Exception as _e:
                    print(f"  [WARN] SS compute failed for candidate (attempt={attempt_val},window={window_val}): {_e}")
                    continue

                # Write back into all_pairs_records for matching entries
                n_updated = update_all_pairs_records_with_ss(
                    target_seq1=str(seq1_t),
                    target_seq2=str(seq2_t),
                    attempt=attempt_val,
                    window=window_val,
                    s1_pct=s1_pct,
                    s2_pct=s2_pct,
                    ss_pred_a=ss_a,
                    ss_pred_b=ss_b,
                    avg_pct=avg_pct
                )

                if n_updated > 0:
                    updated_count += n_updated
                    print(f"  [TOP-ESM SS] Updated {n_updated} record(s): attempt={attempt_val}, window={window_val}, SS_avg={avg_pct:.3f}")
                else:
                    # Best-effort fallback: try to update any matching seq pair regardless of attempt/window
                    fallback_updated = 0
                    for rec in all_pairs_records:
                        if (str(rec.get("translated_integrated_seq_1","")).strip().rstrip("*") == str(seq1_t).strip().rstrip("*")
                            and str(rec.get("translated_integrated_seq_2","")).strip().rstrip("*") == str(seq2_t).strip().rstrip("*")):
                            rec["ss_score_1"] = float(s1_pct)
                            rec["ss_score_2"] = float(s2_pct)
                            rec["ss_score_avg"] = float(avg_pct)
                            rec["ss_pred_1"] = ss_a
                            rec["ss_pred_2"] = ss_b
                            fallback_updated += 1
                    if fallback_updated:
                        updated_count += fallback_updated
                        print(f"  [TOP-ESM SS fallback] Updated {fallback_updated} matching record(s) without exact attempt/window.")
                    else:
                        print(f"  [TOP-ESM SS] No record matched for seq pair (attempt={attempt_val}, window={window_val}); no update performed.")
            except Exception as _e:
                print(f"  [ERROR] Unexpected error evaluating top candidate: {_e}")

        print(f"[TOP-ESM SS] Attempted {attempted_count} top candidates; updated {updated_count} record(s) in all_pairs_records.")
        return updated_count

    # Only run the top-N deferred SS if we deferred SS (DO_SS==False)
    try:
        if (not DO_SS):
            _evaluate_top_n_by_esm_and_compute_ss(top_n=20)
    except Exception as _e:
        print(f"[WARN] Deferred top-N SS pass failed: {_e}")

    # Final reporting & save
    print("\n===== CACHE / EFFICIENCY REPORT =====")
    for k, v in METRICS.items():
        print(f"{k}: {v}")

    if not success:
        print(f"Best achieved (stored metric): seq1={best_overall_scores[0]}, seq2={best_overall_scores[1]}")
        print("SEQ1:", best_overall_seq1)
        print("SEQ2:", best_overall_seq2)
        if best_overall_summaries:
            print(pd.DataFrame(best_overall_summaries).to_string(index=False))

    model_tag = "-".join(str(m) for m in model_numbers)
    out_name = f"length_{model_tag}_row_{row_index_1based}.xlsx"
    out_path = os.path.join(working_directory, out_name)

    # Ensure columns include ss_pred_1/2 even for empty scaffold
    if all_pairs_records:
        pd.DataFrame(all_pairs_records).to_excel(out_path, index=False)
        print(f"\nSaved {len(all_pairs_records)} predicted candidate pairs to {out_path}")
    else:
        pd.DataFrame(columns=[
            "aa_seq_1_original", "aa_seq_2_original",
            "translated_aa_seq_1","translated_aa_seq_2",
            "translated_integrated_seq_1","translated_integrated_seq_2",
            "ss_score_1","ss_score_2","ss_score_avg","ss_pred_1","ss_pred_2",
            "align1","align2","sub1","sub2",
            "esm1","esm2","esm_avg",
            "combined_score","attempt","window","attempt_window"
        ]).to_excel(out_path, index=False)
        print(f"\nNo candidates produced; wrote empty scaffold to {out_path}")

    return out_path
