import numpy as np
import pytest

from eventflow.generation_analysis import (exact_match_rate, lag_counts, repeated_run_lengths, training_windows)


def test_lag_counts_keep_sequence_boundaries():
    sequences = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    assert lag_counts(sequences, 2) == {(0, 2): 1, (1, 3): 1, (3, 1): 1, (2, 0): 1}
    with pytest.raises(ValueError):
        lag_counts(sequences, 4)


def test_exact_windows_include_last_window_but_do_not_join_machines():
    windows, n = training_windows([[0, 1, 2, 3], [3, 2, 1]], 3)
    assert n == 3
    assert exact_match_rate([[0, 1, 2], [1, 2, 3], [3, 2, 1]], windows) == 1
    assert exact_match_rate([[2, 3, 3]], windows) == 0


def test_longest_repeat_run():
    assert repeated_run_lengths([[1, 1, 1, 2], [0, 1, 2, 0], [3, 2, 2, 2]]) == [3, 1, 3]
