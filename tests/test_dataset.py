"""
tests/test_dataset.py
=====================
Unit tests for dataset and tensor generation modules.
Run with: pytest tests/test_dataset.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytest

from src.datasets.sequence_generator import generate_sequences, time_split
from src.datasets.dataset import DroughtDataset


# ── Fixtures ─────────────────────────────────────────────────────────────────

N_NODES    = 10
N_FEATURES = 5
N_TARGETS  = 3
T_STEPS    = 60   # 5 years of monthly data
LOOKBACK   = 12


@pytest.fixture
def dummy_tensors():
    X = np.random.randn(T_STEPS, N_NODES, N_FEATURES).astype(np.float32)
    Y = np.random.randn(T_STEPS, N_NODES, N_TARGETS).astype(np.float32)
    dates = np.array([
        np.datetime64(f"200{i//12:d}-{(i%12)+1:02d}-01")
        for i in range(T_STEPS)
    ])
    return X, Y, dates


@pytest.fixture
def dummy_edge_index():
    src = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    return torch.stack([src, dst], dim=0)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_sequence_shape(dummy_tensors):
    X, Y, dates = dummy_tensors
    X_seq, Y_seq, seq_dates = generate_sequences(X, Y, dates,
                                                  lookback=LOOKBACK, horizon=1)
    expected_samples = T_STEPS - LOOKBACK
    assert X_seq.shape == (expected_samples, LOOKBACK, N_NODES, N_FEATURES)
    assert Y_seq.shape == (expected_samples, N_NODES, N_TARGETS)
    assert len(seq_dates) == expected_samples


def test_sequence_alignment(dummy_tensors):
    """X_seq[t] should contain X[t:t+lookback], Y_seq[t] should be Y[t+lookback]."""
    X, Y, dates = dummy_tensors
    X_seq, Y_seq, _ = generate_sequences(X, Y, dates, lookback=LOOKBACK, horizon=1)
    np.testing.assert_array_equal(X_seq[0], X[:LOOKBACK])
    np.testing.assert_array_equal(Y_seq[0], Y[LOOKBACK])


def test_time_split_no_overlap(dummy_tensors):
    X, Y, dates = dummy_tensors
    X_seq, Y_seq, seq_dates = generate_sequences(X, Y, dates, lookback=LOOKBACK)
    splits = time_split(X_seq, Y_seq, seq_dates,
                        train_end="2003-12", val_end="2004-06")
    train_dates = splits["train"]["dates"]
    val_dates   = splits["val"]["dates"]
    test_dates  = splits["test"]["dates"]
    # No overlap between splits
    assert len(set(train_dates) & set(val_dates))  == 0
    assert len(set(val_dates)   & set(test_dates)) == 0
    assert len(set(train_dates) & set(test_dates)) == 0


def test_dataset_len(dummy_tensors, dummy_edge_index):
    X, Y, dates = dummy_tensors
    X_seq, Y_seq, _ = generate_sequences(X, Y, dates, lookback=LOOKBACK)
    ds = DroughtDataset(X_seq, Y_seq, dummy_edge_index,
                        torch.ones(5, 1, dtype=torch.float32))
    assert len(ds) == len(X_seq)


def test_dataset_item_shapes(dummy_tensors, dummy_edge_index):
    X, Y, dates = dummy_tensors
    X_seq, Y_seq, _ = generate_sequences(X, Y, dates, lookback=LOOKBACK)
    ds   = DroughtDataset(X_seq, Y_seq, dummy_edge_index,
                          torch.ones(5, 1, dtype=torch.float32))
    item = ds[0]
    assert item["X"].shape  == (LOOKBACK, N_NODES, N_FEATURES)
    assert item["Y"].shape  == (N_NODES, N_TARGETS)
    assert item["edge_index"].shape[0] == 2


def test_no_data_leakage(dummy_tensors):
    """Ensure future data is never in an input window."""
    X, Y, dates = dummy_tensors
    X_seq, Y_seq, seq_dates = generate_sequences(X, Y, dates, lookback=LOOKBACK)
    import pandas as pd
    for i in range(len(X_seq)):
        target_date = pd.Timestamp(seq_dates[i])
        input_end   = pd.Timestamp(dates[i + LOOKBACK - 1])
        assert input_end < target_date or input_end == target_date, \
            f"Data leakage at sample {i}: input_end={input_end}, target={target_date}"
