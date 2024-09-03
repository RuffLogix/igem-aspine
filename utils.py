from typing import List

import numpy as np
import numpy.typing as npt

import constants


def valid_sequence(sequence_list: List[str]) -> bool:
    """
    Check if a list of sequences is valid.

    Parameters
    ----------
    sequence_list: List[str]
        A list of sequences, where each sequence is a string of nucleotides (e.g. 'ACGT').

    Returns
    -------
    bool
        True if all sequences are valid, False otherwise.

    Examples
    --------
    >>> valid_sequence(['ACGT', 'ACGT'])
    True

    >>> valid_sequence(['ANGT', 'ACGT', 'ACGT', 'ACGT'])
    False

    >>> valid_sequence(['ACGT', 'ACGT', 'ACGT', 'ACGT', 'ACGT'])
    True
    """

    for sequence in sequence_list:
        for nucleotide in sequence:
            is_n_nucleotide = nucleotide is not "N" and nucleotide is not "n"
            if nucleotide not in constants.SEQUENCE_INDEX and is_n_nucleotide:
                return False

    return True


def encode_sequence(sequence_list: List[str]) -> npt.NDArray[np.float32]:
    """
    Convert a list of sequences into a one-hot encoded tensor.

    Parameters
    ----------
    sequences_list: List[str]
        A list of sequences, where each sequence is a string of nucleotides (e.g. 'ACGT').

    Returns
    -------
    np.ndarray
        A 3D numpy array of shape (n_sequences, sequence_len, 4), where n_sequences is the number of sequences in the input list, and sequence_len is the length of each sequence. The last dimension has size 4 and corresponds to the one-hot encoding of the nucleotides (A, C, G, T).

    Examples
    --------
    >>> encode_sequence(['ACGT'])
    array([[[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]], dtype=float32)
    """

    # Check if the input sequences contain invalid nucleotides
    if not valid_sequence(sequence_list):
        raise ValueError(constants.INVALID_SEQUENCE_NUCLOTIDES_ERROR)
    # Check if the input sequences are empty
    if len(sequence_list) == 0:
        raise ValueError(constants.INVALID_SEQUENCE_EMPTY_ERROR)

    # Get the number of sequences and the length of each sequence
    n_sequences = len(sequence_list)
    sequence_len = len(sequence_list[0])

    # Initialize an array of zeros
    sequence_vector = np.zeros((n_sequences, sequence_len, 4), dtype=np.float32)

    # Fill in the array with the one-hot encoding
    for i in range(n_sequences):
        sequence = sequence_list[i]

        # For each nucleotide in the sequence, set the corresponding one-hot encoding
        for j in range(len(sequence_list[i])):
            if sequence[j] in constants.SEQUENCE_INDEX:
                sequence_vector[i, j, constants.SEQUENCE_INDEX[sequence[j]]] = 1.0

    return sequence_vector
