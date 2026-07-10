"""
evaluate.py
===========
Runs model inference on the test set and computes all metrics.
Saves predictions to outputs/predictions/.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.metrics import compute_all_metrics, classification_metrics
from src.preprocessing.feature_engineering import classify_drought
from src.utils.logger import get_logger

log = get_logger(__name__)

PRED_DIR = Path("outputs/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["SRI_3", "SRI_6", "SRI_12"]


def run_evaluation(model:      nn.Module,
                   loader:     torch.utils.data.DataLoader,
                   edge_index: torch.Tensor,
                   device:     torch.device,
                   run_name:   str = "gcn_gru") -> dict:
    """
    Run inference on test loader, compute metrics, save outputs.
    Returns dict of all metrics keyed by target name.
    """
    model.eval()
    all_preds  = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            X  = batch["X"].to(device)
            Y  = batch["Y"]
            ew = batch.get("edge_weight")
            if ew is not None:
                ew = ew.to(device)
            pred = model(X, edge_index, edge_weight=ew).cpu()
            all_preds.append(pred.numpy())
            all_targets.append(Y.numpy())

    preds   = np.concatenate(all_preds,   axis=0)  # [num_samples, N, 3]
    targets = np.concatenate(all_targets, axis=0)  # [num_samples, N, 3]

    # Save raw predictions
    np.save(PRED_DIR / f"{run_name}_preds.npy",   preds)
    np.save(PRED_DIR / f"{run_name}_targets.npy", targets)
    log.info(f"Predictions saved to {PRED_DIR}")

    # Compute metrics per target
    all_metrics = {}
    for t_idx, t_name in enumerate(TARGET_NAMES):
        obs  = targets[:, :, t_idx].flatten()
        pred = preds[:, :, t_idx].flatten()
        metrics = compute_all_metrics(obs, pred)

        # Drought classification metrics
        obs_class  = classify_drought(pd.Series(obs)).values
        pred_class = classify_drought(pd.Series(pred)).values
        cls_metrics = classification_metrics(obs_class, pred_class)
        metrics.update(cls_metrics)

        all_metrics[t_name] = metrics
        log.info(f"[{t_name}] RMSE={metrics['RMSE']:.4f}  "
                 f"NSE={metrics['NSE']:.4f}  KGE={metrics['KGE']:.4f}  "
                 f"F1={metrics['f1']:.4f}")

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics).T
    out_path   = RESULTS_DIR / f"{run_name}_metrics.csv"
    metrics_df.to_csv(out_path)
    log.info(f"Metrics saved → {out_path}")

    return all_metrics
