"""
build_knn_graph.py
==================
Builds a K-Nearest Neighbour graph from sub-basin centroid coordinates.
Used as a spatial-proximity baseline or fallback when river network
connectivity data is not available.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from src.utils.logger import get_logger

log = get_logger(__name__)
GRAPH_DIR = Path("data/graph")


def build_knn_graph(nodes: pd.DataFrame,
                    k: int = 5,
                    metric: str = "haversine") -> pd.DataFrame:
    """
    Build undirected KNN graph from sub-basin centroid coordinates.

    nodes: DataFrame with columns [subbasin_id, centroid_lat, centroid_lon, node_idx]
    k:     number of nearest neighbours per node
    metric: 'haversine' (geographic) or 'euclidean'

    Returns edge DataFrame: [from_subbasin, to_subbasin, distance_km,
                             src_idx, dst_idx]
    """
    coords = nodes[["centroid_lat", "centroid_lon"]].fillna(0).values

    if metric == "haversine":
        coords_rad = np.radians(coords)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                              metric="haversine")
        nn.fit(coords_rad)
        distances, indices = nn.kneighbors(coords_rad)
        distances = distances * 6371.0    # radians → km
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree",
                              metric="euclidean")
        nn.fit(coords)
        distances, indices = nn.kneighbors(coords)

    id_map  = nodes["subbasin_id"].values
    idx_map = nodes["node_idx"].values

    rows = []
    for i in range(len(nodes)):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            rows.append({
                "from_subbasin": int(id_map[i]),
                "to_subbasin":   int(id_map[j]),
                "distance_km":   float(dist),
                "src_idx":       int(idx_map[i]),
                "dst_idx":       int(idx_map[j]),
            })

    edges = pd.DataFrame(rows)

    out_path = GRAPH_DIR / "knn_edges.csv"
    edges.to_csv(out_path, index=False)
    log.info(f"KNN graph (k={k}): {len(edges)} edges saved → {out_path}")
    return edges
