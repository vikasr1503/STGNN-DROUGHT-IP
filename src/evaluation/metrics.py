"""
metrics.py
==========
Evaluation metrics for hydrological drought prediction.

Regression:     RMSE, MAE, R², NSE (Nash-Sutcliffe), KGE (Kling-Gupta)
Classification: Accuracy, Precision, Recall, F1 (per drought class)
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return float(np.sqrt(mean_squared_error(obs[mask], pred[mask])))


def mae(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return float(mean_absolute_error(obs[mask], pred[mask]))


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    return float(r2_score(obs[mask], pred[mask]))


def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. NSE=1 is perfect."""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[mask], pred[mask]
    return float(1 - np.sum((o - p) ** 2) / (np.sum((o - o.mean()) ** 2) + 1e-8))


def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency. KGE=1 is perfect."""
    mask = ~np.isnan(obs) & ~np.isnan(pred)
    o, p = obs[mask], pred[mask]
    r  = np.corrcoef(o, p)[0, 1]
    alpha = p.std() / (o.std() + 1e-8)
    beta  = p.mean() / (o.mean() + 1e-8)
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))


def classification_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """
    Classification metrics for drought severity classes (0–3).
    obs/pred are integer class labels.
    """
    mask = ~np.isnan(obs.astype(float))
    o = obs[mask].astype(int)
    p = pred[mask].astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        o, p, average="macro", zero_division=0
    )
    return {
        "accuracy":  float(accuracy_score(o, p)),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "confusion_matrix": confusion_matrix(o, p).tolist(),
    }


def compute_all_metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Compute all regression metrics for one target variable."""
    return {
        "RMSE": rmse(obs, pred),
        "MAE":  mae(obs, pred),
        "R2":   r2(obs, pred),
        "NSE":  nse(obs, pred),
        "KGE":  kge(obs, pred),
    }
