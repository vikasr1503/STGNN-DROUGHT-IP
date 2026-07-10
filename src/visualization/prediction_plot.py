"""
prediction_plot.py
==================
Plots observed vs predicted SRI time series for selected sub-basins.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS = Path("results/figures")
RESULTS.mkdir(parents=True, exist_ok=True)


def plot_obs_vs_pred(obs: np.ndarray, pred: np.ndarray,
                     dates: np.ndarray,
                     node_idx: int = 0,
                     target_name: str = "SRI_3",
                     run_name: str = "gcn_gru",
                     save: bool = True) -> None:
    """
    Time series plot of observed vs predicted for one sub-basin node.
    obs/pred: [num_samples, N, num_targets]
    """
    t_map = {"SRI_3": 0, "SRI_6": 1, "SRI_12": 2}
    t_idx = t_map.get(target_name, 0)

    obs_ts  = obs[:, node_idx, t_idx]
    pred_ts = pred[:, node_idx, t_idx]
    dates_dt = pd.to_datetime(dates)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates_dt, obs_ts,  label="Observed", linewidth=1.5, color="royalblue")
    ax.plot(dates_dt, pred_ts, label="Predicted", linewidth=1.5,
            color="tomato", linestyle="--")
    ax.axhline(-0.5, color="orange", linestyle=":", alpha=0.6, label="Mild drought")
    ax.axhline(-1.0, color="red",    linestyle=":", alpha=0.6, label="Moderate drought")
    ax.axhline(-1.5, color="darkred",linestyle=":", alpha=0.6, label="Severe drought")
    ax.fill_between(dates_dt, obs_ts, 0,
                    where=obs_ts < -0.5, alpha=0.15, color="red")
    ax.set_title(f"Sub-basin {node_idx} — {target_name}  ({run_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("SRI")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save:
        path = RESULTS / f"{run_name}_node{node_idx}_{target_name}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_history(history: dict,
                          run_name: str = "gcn_gru",
                          save: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["val_loss"],   label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training History — {run_name}")
    ax.legend()
    plt.tight_layout()

    if save:
        path = RESULTS / f"{run_name}_loss_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
