# blosum62_matrix.py

from Bio import Align
from Bio.Align import substitution_matrices


blosum62_matrix = substitution_matrices.load("BLOSUM62")


# Add symmetric entries for the BLOSUM62 matrix
for (aa1, aa2), score in list(blosum62_matrix.items()):
    blosum62_matrix[(aa2, aa1)] = score

# Define the function to calculate BLOSUM62 similarity
def calculate_blosum62_similarity(seq1, seq2, matrix=blosum62_matrix):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    
    similarity_score = 0
    for aa1, aa2 in zip(seq1, seq2):
        score = matrix.get((aa1, aa2), matrix.get((aa2, aa1), -4))  # Use -4 as a default for unknown pairs
        similarity_score += score
    
    return similarity_score
