"""
build_distance_graph.py
=======================
Builds a distance-threshold graph: connect two sub-basins if their
centroid distance is within a threshold (in km).
Edge weight = inverse distance (closer = stronger connection).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)
GRAPH_DIR = Path("data/graph")
EARTH_R   = 6371.0


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Haversine distance in km between two coordinate pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def build_distance_graph(nodes: pd.DataFrame,
                         threshold_km: float = 100.0,
                         self_loops:   bool  = False) -> pd.DataFrame:
    """
    Connect sub-basins whose centroids are within threshold_km of each other.

    nodes: DataFrame with [subbasin_id, centroid_lat, centroid_lon, node_idx]
    threshold_km: maximum centroid distance for an edge to exist
    self_loops:   whether to include self-loop edges

    Returns edge DataFrame: [from_subbasin, to_subbasin, distance_km,
                             weight, src_idx, dst_idx]
    """
    n = len(nodes)
    rows = []

    for i in range(n):
        for j in range(n):
            if i == j:
                if self_loops:
                    rows.append({
                        "from_subbasin": int(nodes["subbasin_id"].iloc[i]),
                        "to_subbasin":   int(nodes["subbasin_id"].iloc[i]),
                        "distance_km":   0.0,
                        "weight":        1.0,
                        "src_idx":       int(nodes["node_idx"].iloc[i]),
                        "dst_idx":       int(nodes["node_idx"].iloc[i]),
                    })
                continue

            lat1 = nodes["centroid_lat"].iloc[i]
            lon1 = nodes["centroid_lon"].iloc[i]
            lat2 = nodes["centroid_lat"].iloc[j]
            lon2 = nodes["centroid_lon"].iloc[j]

            if any(pd.isna([lat1, lon1, lat2, lon2])):
                continue

            dist = haversine_distance(lat1, lon1, lat2, lon2)
            if dist <= threshold_km:
                rows.append({
                    "from_subbasin": int(nodes["subbasin_id"].iloc[i]),
                    "to_subbasin":   int(nodes["subbasin_id"].iloc[j]),
                    "distance_km":   float(dist),
                    "weight":        float(1.0 / (dist + 1e-6)),
                    "src_idx":       int(nodes["node_idx"].iloc[i]),
                    "dst_idx":       int(nodes["node_idx"].iloc[j]),
                })

    edges = pd.DataFrame(rows)
    out_path = GRAPH_DIR / "distance_edges.csv"
    edges.to_csv(out_path, index=False)
    log.info(f"Distance graph (threshold={threshold_km}km): "
             f"{len(edges)} edges → {out_path}")
    return edges
