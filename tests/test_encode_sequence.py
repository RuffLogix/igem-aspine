import numpy as np

import constants
from utils import encode_sequence

ENCODE_SEQUENCE_TESTCASES = [
    # Testcase 1. Single sequence.
    (
        ["ACGT"],
        np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        ),
        False,
    ),
    # Testcase 2. Empty sequence.
    ([], constants.INVALID_SEQUENCE_EMPTY_ERROR, True),
    # Testcase 3. Invalid nucleotides in sequence.
    (["AcZT"], constants.INVALID_SEQUENCE_NUCLOTIDES_ERROR, True),
    # Testcase 4. Multiple sequences.
    (
        ["ACCC", "GT"],
        np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=np.float32,
        ),
        False,
    ),
    # Testcase 5. Single Sequence with N.
    (
        ["ACNT"],
        np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        ),
        False,
    ),
]


def test_encode_sequence():
    for sequence_list, expected, is_error in ENCODE_SEQUENCE_TESTCASES:
        if is_error:
            try:
                print(encode_sequence(sequence_list), expected)
            except Exception as e:
                assert str(e) == expected
        else:
            assert np.array_equal(encode_sequence(sequence_list), expected)
