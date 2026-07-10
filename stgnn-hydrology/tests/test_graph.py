"""
tests/test_graph.py
===================
Unit tests for graph construction modules.
Run with: pytest tests/test_graph.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_nodes():
    return pd.DataFrame({
        "subbasin_id":   list(range(10)),
        "node_idx":      list(range(10)),
        "area_m2":       [1e5] * 10,
        "area_km2":      [0.1] * 10,
        "centroid_lat":  [16.0 + i * 0.1 for i in range(10)],
        "centroid_lon":  [76.0 + i * 0.1 for i in range(10)],
        "elevation_m":   [500.0 + i * 10 for i in range(10)],
    })


@pytest.fixture
def dummy_edges(dummy_nodes):
    rows = []
    for i in range(9):
        rows.append({
            "from_subbasin": i,
            "to_subbasin":   i + 1,
            "src_idx":       i,
            "dst_idx":       i + 1,
            "river_length_km": 10.0,
        })
    return pd.DataFrame(rows)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_node_table_shape(dummy_nodes):
    assert len(dummy_nodes) == 10
    assert "subbasin_id" in dummy_nodes.columns
    assert "node_idx" in dummy_nodes.columns


def test_edge_list_directed(dummy_edges):
    """Edges should be directed (no automatic reverse)."""
    assert len(dummy_edges) == 9
    reverse = dummy_edges.rename(columns={
        "from_subbasin": "to_subbasin",
        "to_subbasin": "from_subbasin"
    })
    merged = dummy_edges.merge(reverse, on=["from_subbasin", "to_subbasin"])
    assert len(merged) == 0, "Unexpected reverse edges found"


def test_static_adjacency(dummy_edges):
    from src.graph.adjacency import build_static_adjacency
    edge_index, edge_attr = build_static_adjacency(
        dummy_edges, n_nodes=10, self_loops=False
    )
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == len(dummy_edges)
    assert edge_index.dtype == torch.long


def test_adjacency_with_self_loops(dummy_edges):
    from src.graph.adjacency import build_static_adjacency
    ei, ea = build_static_adjacency(dummy_edges, n_nodes=10, self_loops=True)
    assert ei.shape[1] == len(dummy_edges) + 10   # edges + self-loops


def test_knn_graph_edge_count(dummy_nodes):
    from src.graph.build_knn_graph import build_knn_graph
    k = 3
    edges = build_knn_graph(dummy_nodes, k=k, metric="euclidean")
    # Each of 10 nodes has k outgoing edges
    assert len(edges) == 10 * k


def test_adjacency_matrix_shape(dummy_edges):
    from src.graph.adjacency import build_adjacency_matrix
    A = build_adjacency_matrix(dummy_edges, n_nodes=10)
    assert A.shape == (10, 10)
    assert A.dtype == np.float32
    # Diagonal should be 0 (no self-loops in dummy_edges)
    assert A.trace() == 0.0
