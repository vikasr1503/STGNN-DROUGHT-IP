"""
scripts/evaluate.py
===================
Load a trained model checkpoint and evaluate on the test set.

Usage:
  python scripts/evaluate.py --run_name gcn_gru
  python scripts/evaluate.py --run_name gat_gru --model gat_gru
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers  import load_config
from src.utils.seed     import set_seed
from src.utils.logger   import get_logger

from src.graph.build_nodes  import build_node_table
from src.graph.build_edges  import build_edge_list_from_network
from src.graph.adjacency    import build_static_adjacency

from src.datasets.tensor_builder     import build_tensors
from src.datasets.sequence_generator import generate_sequences, time_split
from src.datasets.dataset            import make_dataloaders

from src.models.gcn_gru  import DynamicDirectedSTGNN
from src.models.gat_gru  import GATGRUModel

from src.evaluation.evaluate          import run_evaluation
from src.visualization.prediction_plot import plot_obs_vs_pred

log = get_logger(__name__, log_file="outputs/logs/evaluate.log")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",   required=True)
    parser.add_argument("--model",      default="gcn_gru",
                        choices=["gcn_gru", "gat_gru", "graph_wavenet"])
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg_dir   = Path(args.config_dir)
    cfg_train = load_config(cfg_dir / "train.yaml")
    cfg_model = load_config(cfg_dir / "model.yaml")
    cfg_graph = load_config(cfg_dir / "graph.yaml")

    set_seed(cfg_train["training"]["seed"])

    device_str = args.device or cfg_train["training"]["device"]
    device = torch.device(device_str if torch.cuda.is_available()
                          or device_str == "cpu" else "cpu")

    log.info(f"Evaluating: {args.run_name}  device={device}")

    # ── Load graph ────────────────────────────────────────────────────────────
    nodes      = build_node_table()
    edges      = build_edge_list_from_network()
    edge_index, edge_attr = build_static_adjacency(
        edges, n_nodes=cfg_graph["graph"]["n_nodes"]
    )
    edge_index = edge_index.to(device)

    # ── Load tensors ──────────────────────────────────────────────────────────
    X     = np.load("data/tensors/X.npy")
    Y     = np.load("data/tensors/Y.npy")
    dates = np.load("data/tensors/dates.npy", allow_pickle=True)

    t_cfg   = cfg_train["training"]
    X_seq, Y_seq, seq_dates = generate_sequences(
        X, Y, dates,
        lookback=t_cfg["lookback"], horizon=t_cfg["horizon"]
    )
    splits  = time_split(X_seq, Y_seq, seq_dates,
                         train_end=t_cfg["split"]["train_end"],
                         val_end=t_cfg["split"]["val_end"])
    loaders = make_dataloaders(splits, edge_index.cpu(), edge_attr,
                               batch_size=t_cfg["batch_size"])

    # ── Load model ────────────────────────────────────────────────────────────
    F     = X.shape[-1]
    m_cfg = cfg_model["model"]
    if args.model == "gcn_gru":
        model = DynamicDirectedSTGNN(
            in_channels=F,
            gcn_hidden=m_cfg["gcn"]["hidden_channels"],
            gcn_layers=m_cfg["gcn"]["num_layers"],
            gru_hidden=m_cfg["gru"]["hidden_size"],
            gru_layers=m_cfg["gru"]["num_layers"],
            num_targets=len(m_cfg["output"]["targets"]),
        )
    else:
        model = GATGRUModel(
            in_channels=F,
            gat_hidden=m_cfg["gcn"]["hidden_channels"],
            gru_hidden=m_cfg["gru"]["hidden_size"],
            gru_layers=m_cfg["gru"]["num_layers"],
            num_targets=len(m_cfg["output"]["targets"]),
        )

    ckpt = Path(t_cfg["save_dir"]) / f"{args.run_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    metrics = run_evaluation(model, loaders["test"],
                             edge_index, device, args.run_name)

    # Visualise 3 sub-basins
    preds   = np.load(f"outputs/predictions/{args.run_name}_preds.npy")
    targets = np.load(f"outputs/predictions/{args.run_name}_targets.npy")
    for ni in range(min(3, preds.shape[1])):
        for tgt in ["SRI_3", "SRI_6", "SRI_12"]:
            plot_obs_vs_pred(targets, preds, splits["test"]["dates"],
                             node_idx=ni, target_name=tgt,
                             run_name=args.run_name)

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()
