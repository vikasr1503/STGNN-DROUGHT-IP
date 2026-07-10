"""
scripts/train.py
================
Main entry point for training the STGNN model.

Usage:
  python scripts/train.py
  python scripts/train.py --model gat_gru --run_name experiment_01
  python scripts/train.py --model gcn_gru --device cpu
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# ── project imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed     import set_seed
from src.utils.helpers  import load_config
from src.utils.logger   import get_logger

from src.preprocessing.load_data          import (load_subbasin_table, load_discharge,
                                                   load_swat_outputs, load_groundwater,
                                                   load_reservoir)
from src.preprocessing.preprocess         import run_preprocessing
from src.preprocessing.feature_engineering import compute_all_indices, build_node_features

from src.graph.build_nodes   import build_node_table
from src.graph.build_edges   import build_edge_list_from_network
from src.graph.adjacency     import build_static_adjacency

from src.datasets.tensor_builder    import build_tensors
from src.datasets.sequence_generator import generate_sequences, time_split
from src.datasets.dataset            import make_dataloaders

from src.models.gcn_gru  import DynamicDirectedSTGNN
from src.models.gat_gru  import GATGRUModel

from src.training.trainer import train

from src.evaluation.evaluate          import run_evaluation
from src.visualization.prediction_plot import plot_training_history, plot_obs_vs_pred

log = get_logger(__name__, log_file="outputs/logs/train.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="gcn_gru",
                        choices=["gcn_gru", "gat_gru"])
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device",   default=None)
    parser.add_argument("--config_dir", default="configs")
    args = parser.parse_args()

    # ── Load configs ─────────────────────────────────────────────────────────
    cfg_dir   = Path(args.config_dir)
    cfg_train = load_config(cfg_dir / "train.yaml")
    cfg_model = load_config(cfg_dir / "model.yaml")
    cfg_graph = load_config(cfg_dir / "graph.yaml")

    run_name = args.run_name or args.model
    set_seed(cfg_train["training"]["seed"])

    device_str = args.device or cfg_train["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available()
                          or device_str == "cpu" else "cpu")
    log.info(f"Device: {device}  |  Model: {args.model}  |  Run: {run_name}")

    # ── STEP 1: Load data ────────────────────────────────────────────────────
    log.info("Loading data sources ...")
    subbasins   = load_subbasin_table()
    discharge   = load_discharge()
    swat        = load_swat_outputs()
    groundwater = load_groundwater()
    reservoir   = load_reservoir()

    # ── STEP 2: Preprocess & merge ───────────────────────────────────────────
    log.info("Preprocessing ...")
    period     = cfg_train["training"]["period"]
    merged     = run_preprocessing(
        discharge, swat, groundwater, reservoir,
        subbasin_ids=subbasins["subbasin_id"].tolist(),
        start=period["start"], end=period["end"]
    )

    # ── STEP 3: Drought indices ──────────────────────────────────────────────
    log.info("Computing drought indices ...")
    di_cfg = cfg_train["drought_indices"]
    merged = compute_all_indices(
        merged,
        spi_scales=di_cfg["spi"],
        sri_scales=di_cfg["sri"],
        ssi_scales=di_cfg["ssi"],
        gwi_scales=di_cfg["gwi"],
    )

    # ── STEP 4: Build graph ──────────────────────────────────────────────────
    log.info("Building graph ...")
    nodes      = build_node_table()
    edges      = build_edge_list_from_network()
    edge_index, edge_attr = build_static_adjacency(
        edges, n_nodes=cfg_graph["graph"]["n_nodes"]
    )
    edge_index = edge_index.to(device)

    # ── STEP 5: Build tensors ────────────────────────────────────────────────
    log.info("Building feature tensors ...")
    static_cols = ["subbasin_id", "elevation_m", "area_km2",
                   "slope_pct", "drainage_density"]
    static_df = nodes[[c for c in static_cols if c in nodes.columns]]
    features  = build_node_features(merged, static_df)
    X, Y, dates = build_tensors(
        features, nodes, start=period["start"], end=period["end"]
    )

    # ── STEP 6: Sequences & splits ───────────────────────────────────────────
    log.info("Generating sequences ...")
    t_cfg      = cfg_train["training"]
    X_seq, Y_seq, seq_dates = generate_sequences(
        X, Y, dates.values,
        lookback=t_cfg["lookback"], horizon=t_cfg["horizon"]
    )
    splits = time_split(
        X_seq, Y_seq, seq_dates,
        train_end=t_cfg["split"]["train_end"],
        val_end=t_cfg["split"]["val_end"],
    )
    loaders = make_dataloaders(splits, edge_index.cpu(), edge_attr,
                               batch_size=t_cfg["batch_size"])

    # ── STEP 7: Build model ──────────────────────────────────────────────────
    log.info(f"Building model: {args.model} ...")
    m_cfg = cfg_model["model"]
    F     = X.shape[-1]
    if args.model == "gcn_gru":
        model = DynamicDirectedSTGNN(
            in_channels=F,
            gcn_hidden=m_cfg["gcn"]["hidden_channels"],
            gcn_layers=m_cfg["gcn"]["num_layers"],
            gru_hidden=m_cfg["gru"]["hidden_size"],
            gru_layers=m_cfg["gru"]["num_layers"],
            num_targets=len(m_cfg["output"]["targets"]),
            dropout=m_cfg["gcn"]["dropout"],
        )
    else:
        model = GATGRUModel(
            in_channels=F,
            gat_hidden=m_cfg["gcn"]["hidden_channels"],
            gru_hidden=m_cfg["gru"]["hidden_size"],
            gru_layers=m_cfg["gru"]["num_layers"],
            num_targets=len(m_cfg["output"]["targets"]),
            dropout=m_cfg["gcn"]["dropout"],
        )

    # ── STEP 8: Train ────────────────────────────────────────────────────────
    log.info("Training ...")
    history = train(model, loaders, cfg_train, edge_index, device, run_name)
    plot_training_history(history, run_name=run_name)

    # ── STEP 9: Evaluate ─────────────────────────────────────────────────────
    log.info("Evaluating on test set ...")
    # Load best checkpoint
    ckpt_path = Path(t_cfg["save_dir"]) / f"{run_name}_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    metrics = run_evaluation(model, loaders["test"], edge_index, device, run_name)

    # Plot predictions for first 3 sub-basins
    preds   = np.load(f"outputs/predictions/{run_name}_preds.npy")
    targets = np.load(f"outputs/predictions/{run_name}_targets.npy")
    for node_idx in range(min(3, preds.shape[1])):
        plot_obs_vs_pred(targets, preds, splits["test"]["dates"],
                         node_idx=node_idx, run_name=run_name)

    log.info("All done.")
    log.info(json.dumps(
        {k: {m: round(v, 4) for m, v in mv.items()
             if isinstance(v, float)}
         for k, mv in metrics.items()},
        indent=2
    ))


if __name__ == "__main__":
    main()
