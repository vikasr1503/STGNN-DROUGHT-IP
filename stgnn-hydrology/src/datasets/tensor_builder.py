"""
tensor_builder.py
=================
Builds the spatio-temporal tensor X of shape [T, N, F] and
target tensor Y of shape [T, N, num_targets].

T = number of monthly timesteps
N = 130 sub-basins (nodes)
F = number of node features

Saves:
  data/tensors/X.npy        — feature tensor [T, N, F]
  data/tensors/Y.npy        — target tensor  [T, N, 3]  (SRI-3, SRI-6, SRI-12)
  data/tensors/dates.npy    — datetime array [T]
  data/tensors/metadata.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

TENSORS = Path("data/tensors")
TENSORS.mkdir(parents=True, exist_ok=True)

# Dynamic node features in fixed order
DYNAMIC_FEATURES = [
    "rainfall_mm", "runoff_mm", "soil_moisture_mm", "et_mm",
    "groundwater_mm", "streamflow_m3s", "temperature_c",
    "release_mcm", "enso_index", "spi_3", "sri_3",
]

# Static node features (appended after dynamic)
STATIC_FEATURES = ["elevation_m", "area_km2", "slope_pct", "drainage_density"]

# Prediction targets
TARGETS = ["sri_3", "sri_6", "sri_12"]


def build_tensors(features: pd.DataFrame,
                  nodes: pd.DataFrame,
                  start: str = "1970-01",
                  end: str   = "2025-12") -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Pivot the tidy (subbasin_id, date, feature) DataFrame into [T, N, F] tensor.

    features: output of feature_engineering.build_node_features()
    nodes:    output of build_nodes.build_node_table()
    """
    date_range  = pd.date_range(start=start, end=end, freq="MS")
    subbasin_ids = nodes.sort_values("node_idx")["subbasin_id"].tolist()
    N = len(subbasin_ids)
    T = len(date_range)

    features = features.copy()
    features["date"] = pd.to_datetime(features["date"]).dt.to_period("M").dt.to_timestamp()

    # Available dynamic features
    dyn_cols = [c for c in DYNAMIC_FEATURES if c in features.columns]
    sta_cols = [c for c in STATIC_FEATURES  if c in features.columns]
    F = len(dyn_cols) + len(sta_cols)

    log.info(f"Building tensor: T={T}, N={N}, F={F}  "
             f"(dyn={len(dyn_cols)}, static={len(sta_cols)})")

    X = np.full((T, N, F), np.nan, dtype=np.float32)
    Y = np.full((T, N, len(TARGETS)), np.nan, dtype=np.float32)

    feat_indexed = features.set_index(["date", "subbasin_id"])
    tgt_cols = [c for c in TARGETS if c in features.columns]

    for t, date in enumerate(date_range):
        for n, sb_id in enumerate(subbasin_ids):
            key = (date, sb_id)
            if key not in feat_indexed.index:
                continue

            row = feat_indexed.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Dynamic features
            for f_idx, col in enumerate(dyn_cols):
                X[t, n, f_idx] = row.get(col, np.nan)

            # Static features (same for all t, but stored in tensor for convenience)
            offset = len(dyn_cols)
            for f_idx, col in enumerate(sta_cols):
                X[t, n, offset + f_idx] = row.get(col, np.nan)

            # Targets
            for y_idx, col in enumerate(tgt_cols):
                Y[t, n, y_idx] = row.get(col, np.nan)

    # Save
    np.save(TENSORS / "X.npy",     X)
    np.save(TENSORS / "Y.npy",     Y)
    np.save(TENSORS / "dates.npy", date_range.values)

    metadata = {
        "T": T, "N": N, "F": F,
        "dynamic_features": dyn_cols,
        "static_features":  sta_cols,
        "targets":          tgt_cols,
        "start": str(start), "end": str(end),
        "subbasin_ids": subbasin_ids,
    }
    with open(TENSORS / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"X.npy: {X.shape}  Y.npy: {Y.shape}")
    return X, Y, date_range
