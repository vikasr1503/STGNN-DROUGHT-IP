"""
scripts/predict.py
==================
Run inference for a specific date or date range.

Usage:
  python scripts/predict.py --run_name gcn_gru --date 2023-06
  python scripts/predict.py --run_name gcn_gru --start 2023-01 --end 2023-12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import load_config
from src.utils.logger  import get_logger
from src.graph.build_nodes  import build_node_table
from src.graph.build_edges  import build_edge_list_from_network
from src.graph.adjacency    import build_static_adjacency
from src.models.gcn_gru     import DynamicDirectedSTGNN

log = get_logger(__name__)

TARGET_NAMES = ["SRI_3", "SRI_6", "SRI_12"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",   required=True)
    parser.add_argument("--model",      default="gcn_gru")
    parser.add_argument("--date",       default=None,
                        help="Single prediction date e.g. 2023-06")
    parser.add_argument("--start",      default=None)
    parser.add_argument("--end",        default=None)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    cfg_dir   = Path(args.config_dir)
    cfg_train = load_config(cfg_dir / "train.yaml")
    cfg_model = load_config(cfg_dir / "model.yaml")
    cfg_graph = load_config(cfg_dir / "graph.yaml")

    device = torch.device(args.device)

    # ── Load pre-built tensors ────────────────────────────────────────────────
    X     = np.load("data/tensors/X.npy")
    Y     = np.load("data/tensors/Y.npy")
    dates = np.load("data/tensors/dates.npy", allow_pickle=True)
    dates_dt = pd.to_datetime(dates)

    lookback = cfg_train["training"]["lookback"]

    # Determine target date(s)
    if args.date:
        target_dates = [pd.Timestamp(args.date)]
    elif args.start and args.end:
        target_dates = pd.date_range(args.start, args.end, freq="MS").tolist()
    else:
        target_dates = [dates_dt[-1]]

    # ── Load model ────────────────────────────────────────────────────────────
    nodes = build_node_table()
    edges = build_edge_list_from_network()
    edge_index, _ = build_static_adjacency(
        edges, n_nodes=cfg_graph["graph"]["n_nodes"]
    )
    edge_index = edge_index.to(device)

    F     = X.shape[-1]
    m_cfg = cfg_model["model"]
    model = DynamicDirectedSTGNN(
        in_channels=F,
        gcn_hidden=m_cfg["gcn"]["hidden_channels"],
        gcn_layers=m_cfg["gcn"]["num_layers"],
        gru_hidden=m_cfg["gru"]["hidden_size"],
        gru_layers=m_cfg["gru"]["num_layers"],
        num_targets=len(m_cfg["output"]["targets"]),
    )
    ckpt = Path(cfg_train["training"]["save_dir"]) / f"{args.run_name}_best.pth"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    # ── Predict ───────────────────────────────────────────────────────────────
    out_dir = Path("outputs/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    with torch.no_grad():
        for target_date in target_dates:
            t_idx = np.searchsorted(dates_dt, target_date)
            if t_idx < lookback:
                log.warning(f"Not enough history before {target_date}. Skipping.")
                continue

            x_window = X[t_idx - lookback: t_idx]           # [lookback, N, F]
            x_tensor = torch.tensor(x_window, dtype=torch.float32)
            x_tensor = x_tensor.unsqueeze(0).to(device)     # [1, lookback, N, F]

            pred = model(x_tensor, edge_index).squeeze(0).cpu().numpy()  # [N, 3]

            for ni, sb_id in enumerate(nodes["subbasin_id"].tolist()):
                row = {"date": target_date.strftime("%Y-%m"), "subbasin_id": sb_id}
                for ti, tname in enumerate(TARGET_NAMES):
                    row[tname] = round(float(pred[ni, ti]), 4)
                all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    out_path = out_dir / f"{args.run_name}_predictions.csv"
    results_df.to_csv(out_path, index=False)
    log.info(f"Predictions saved → {out_path}")
    print(results_df.head(10).to_string())


if __name__ == "__main__":
    main()
