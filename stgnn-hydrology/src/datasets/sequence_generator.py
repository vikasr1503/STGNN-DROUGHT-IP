"""
sequence_generator.py
=====================
Generates sliding-window input/output sequences for STGNN training.

For each timestep t:
  Input  X_seq[t] : X[t - lookback : t]  → shape [lookback, N, F]
  Target Y_seq[t] : Y[t]                 → shape [N, num_targets]

Also handles train/val/test splits by time.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

TENSORS = Path("data/tensors")


def generate_sequences(X: np.ndarray,
                       Y: np.ndarray,
                       dates: np.ndarray,
                       lookback: int = 12,
                       horizon:  int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slide a window of `lookback` months across the time axis.

    Returns:
      X_seq: [num_samples, lookback, N, F]
      Y_seq: [num_samples, N, num_targets]
      seq_dates: [num_samples]  — date of the prediction target
    """
    T = X.shape[0]
    num_samples = T - lookback - horizon + 1

    if num_samples <= 0:
        raise ValueError(f"Not enough timesteps: T={T}, lookback={lookback}, horizon={horizon}")

    X_seq     = np.stack([X[i : i + lookback]        for i in range(num_samples)])
    Y_seq     = np.stack([Y[i + lookback + horizon - 1] for i in range(num_samples)])
    seq_dates = np.array([dates[i + lookback + horizon - 1] for i in range(num_samples)])

    log.info(f"Sequences: X_seq {X_seq.shape}, Y_seq {Y_seq.shape}")
    return X_seq, Y_seq, seq_dates


def time_split(X_seq: np.ndarray,
               Y_seq: np.ndarray,
               seq_dates: np.ndarray,
               train_end: str  = "2015-12",
               val_end:   str  = "2019-12") -> dict:
    """
    Split sequences into train / val / test by time.
    Returns dict with keys: train, val, test.
    Each value is a dict with keys: X, Y, dates.
    """
    dates_dt = pd.to_datetime(seq_dates)
    train_mask = dates_dt <= pd.Timestamp(train_end)
    val_mask   = (dates_dt > pd.Timestamp(train_end)) & (dates_dt <= pd.Timestamp(val_end))
    test_mask  = dates_dt > pd.Timestamp(val_end)

    splits = {}
    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        splits[name] = {
            "X": X_seq[mask],
            "Y": Y_seq[mask],
            "dates": seq_dates[mask],
        }
        log.info(f"  {name}: {mask.sum()} samples")

    return splits
