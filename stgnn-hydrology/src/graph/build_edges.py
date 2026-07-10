"""
build_edges.py
==============
Constructs directed edges for the Krishna River Basin sub-basin graph
using the actual GIS-derived river network topology from Channels_KB.csv.

Edge direction: LINKNO → DSLINKNO  (upstream reach → downstream reach)

Channels_KB.csv columns used:
    LINKNO      — unique ID of the current river reach
    DSLINKNO    — ID of the immediate downstream reach (-1 = outlet)
    USLINKNO1   — first upstream reach flowing into this reach
    USLINKNO2   — second upstream reach at confluences (-1 if none)
    DSNODEID    — downstream junction/node identifier
    Length      — river reach length (m)
    Slope       — channel slope
    strmOrder   — Strahler stream order
    Magnitude   — tributary magnitude
    DSContArea  — downstream contributing area
    USContArea  — upstream contributing area
    strmDrop    — elevation drop along reach
    StraightL   — straight-line reach length

Output:
    data/graph/edges.csv            — full directed edge list
    data/graph/edge_topology.csv    — confluence summary (USLINKNO1/2)
    data/graph/graph_stats.txt      — graph validation report
"""

from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch

from src.utils.logger import get_logger

log       = get_logger(__name__)
GRAPH_DIR = Path("data/graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Columns used as static edge attributes in the GNN
EDGE_ATTR_COLS = [
    "Length",
    "Slope",
    "strmOrder",
    "Magnitude",
    "DSContArea",
    "USContArea",
    "strmDrop",
    "StraightL",
]

# DSLINKNO = -1 means this reach flows to the basin outlet (no downstream)
OUTLET_SENTINEL = -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_edge_list_from_network(
    channel_csv: str  = "data/raw/Channels_KB.csv",
    nodes_csv:   str  = "data/graph/nodes.csv",
    linkno_to_subbasin_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the directed edge list from the GIS river channel network.

    Parameters
    ----------
    channel_csv             : path to Channels_KB.csv
    nodes_csv               : path to nodes.csv (output of build_nodes.py)
    linkno_to_subbasin_csv  : optional CSV mapping LINKNO → subbasin_id
                              Columns: linkno, subbasin_id
                              If not provided, LINKNO is used directly as
                              the node identifier (requires LINKNO values
                              to match subbasin_id values).

    Returns
    -------
    DataFrame with one row per directed edge, including:
        LINKNO, DSLINKNO, src_idx, dst_idx, edge attributes
    """
    log.info("=" * 55)
    log.info("Building edge list from river network topology")
    log.info("=" * 55)

    # ── Load inputs ─────────────────────────────────────────────
    channels = _load_channels(channel_csv)
    nodes    = pd.read_csv(nodes_csv)

    # ── Build LINKNO → subbasin_id mapping ──────────────────────
    linkno_map = _build_linkno_mapping(channels, nodes, linkno_to_subbasin_csv)

    # ── Construct directed edges from LINKNO → DSLINKNO ─────────
    edges = _build_directed_edges(channels, linkno_map)

    # ── Attach edge attributes from channel table ────────────────
    edges = _attach_edge_attributes(edges, channels)

    # ── Map subbasin_id → node_idx ───────────────────────────────
    edges = _map_to_node_indices(edges, nodes)

    # ── Save topology summary (confluences) ──────────────────────
    _save_confluence_summary(channels)

    # ── Validate the graph ───────────────────────────────────────
    _validate_graph(edges, nodes)

    # ── Save edges ───────────────────────────────────────────────
    out_path = GRAPH_DIR / "edges.csv"
    edges.to_csv(out_path, index=False)
    log.info(f"Edge list saved: {len(edges)} edges → {out_path}")

    return edges


def compute_dynamic_edge_weights(
    edges:    pd.DataFrame,
    features: pd.DataFrame,
    date:     pd.Timestamp,
) -> np.ndarray:
    """
    Compute time-varying edge weights for a single timestep t.

    Formula:
        W(A→B, t) = w1 * Runoff_A(t)
                  + w2 * SPI_A(t)
                  + w3 * sin(2π * month/12)     [seasonal signal]
                  + w4 * ReservoirRelease_A(t)
                  + w5 * (1 / Length_AB)         [inverse travel distance]

    Each weight is optional — if the feature column is not in `features`,
    its contribution is set to 0 without error. This allows the same
    function to work as new datasets (groundwater, reservoir) are added.

    Parameters
    ----------
    edges    : edge DataFrame with LINKNO, src_subbasin columns
    features : node feature DataFrame for one timestep
               columns: subbasin_id, runoff_mm, spi_3, release_mcm, ...
    date     : the timestep being processed

    Returns
    -------
    np.ndarray of shape [num_edges]
    """
    feat_t = (
        features[features["date"] == date]
        .set_index("subbasin_id")
    )

    month  = date.month
    season = np.sin(2 * np.pi * month / 12)

    # Weights — adjust as more data sources become available
    w = {
        "runoff_mm":   0.30,
        "spi_3":       0.20,
        "season":      0.10,
        "release_mcm": 0.25,
        "inv_length":  0.15,
    }

    weights = []

    for _, edge in edges.iterrows():
        src = edge.get("src_subbasin", edge.get("LINKNO"))

        def feat(col: str, default: float = 0.0) -> float:
            if col not in feat_t.columns:
                return default
            if src not in feat_t.index:
                return default
            v = feat_t.loc[src, col]
            return float(v) if not pd.isna(v) else default

        length_m   = edge.get("Length", 1.0) or 1.0
        inv_length = 1.0 / (length_m + 1e-6)

        weight = (
            w["runoff_mm"]   * feat("runoff_mm")
            + w["spi_3"]     * feat("spi_3")
            + w["season"]    * season
            + w["release_mcm"] * feat("release_mcm")
            + w["inv_length"] * inv_length
        )
        weights.append(weight)

    return np.array(weights, dtype=np.float32)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_channels(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Channels_KB.csv not found: {p}\n"
            f"Place Channels_KB.csv (from advisor GIS outputs) at: {p}"
        )

    df = pd.read_csv(p)
    log.info(f"Loaded {p.name}: {len(df)} river reaches")
    log.info(f"Columns: {list(df.columns)}")

    # Required topology columns
    required = {"LINKNO", "DSLINKNO"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Channels_KB.csv is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Duplicate LINKNO check
    dupes = df["LINKNO"].duplicated()
    if dupes.any():
        raise ValueError(
            f"Duplicate LINKNO values found: {df.loc[dupes, 'LINKNO'].tolist()}"
        )

    log.info(
        f"  LINKNO range   : {df['LINKNO'].min()} → {df['LINKNO'].max()}"
    )
    log.info(
        f"  Outlet reaches : {(df['DSLINKNO'] == OUTLET_SENTINEL).sum()} "
        f"(DSLINKNO = {OUTLET_SENTINEL})"
    )

    # Log which edge attribute columns are available
    available_attrs = [c for c in EDGE_ATTR_COLS if c in df.columns]
    missing_attrs   = [c for c in EDGE_ATTR_COLS if c not in df.columns]
    log.info(f"  Edge attributes available : {available_attrs}")
    if missing_attrs:
        log.warning(f"  Edge attributes missing   : {missing_attrs} (will be NaN)")

    return df


def _build_linkno_mapping(
    channels: pd.DataFrame,
    nodes:    pd.DataFrame,
    mapping_csv: Optional[str],
) -> dict:
    """
    Build { LINKNO: subbasin_id } mapping.

    If a mapping CSV is provided, load it directly.
    Otherwise, assume LINKNO values can be matched to subbasin_id directly
    (valid when SWAT outputs use the same numbering for reaches and sub-basins).
    """
    if mapping_csv and Path(mapping_csv).exists():
        mapping_df = pd.read_csv(mapping_csv)

        required = {"linkno", "subbasin_id"}
        missing  = required - set(mapping_df.columns)
        if missing:
            raise ValueError(
                f"Mapping CSV must have columns: {required}. "
                f"Missing: {missing}"
            )

        linkno_map = dict(
            zip(mapping_df["linkno"], mapping_df["subbasin_id"])
        )
        log.info(
            f"Loaded LINKNO→subbasin mapping: {len(linkno_map)} entries "
            f"from {mapping_csv}"
        )

    else:
        # Direct mapping: LINKNO == subbasin_id
        # Common in SWAT outputs where reach i drains sub-basin i
        valid_ids  = set(nodes["subbasin_id"].tolist())
        all_linknos = set(channels["LINKNO"].tolist())
        overlap    = all_linknos & valid_ids

        linkno_map = {lid: lid for lid in overlap}

        unmatched = all_linknos - valid_ids
        if unmatched:
            log.warning(
                f"{len(unmatched)} LINKNO values have no matching subbasin_id "
                f"and will be excluded from the graph.\n"
                f"  Sample unmatched: {sorted(unmatched)[:10]}\n"
                f"  Provide a linkno_to_subbasin_csv mapping to resolve this."
            )
        log.info(
            f"Direct LINKNO=subbasin_id mapping: "
            f"{len(linkno_map)} matched, {len(unmatched)} unmatched"
        )

    return linkno_map


def _build_directed_edges(
    channels:   pd.DataFrame,
    linkno_map: dict,
) -> pd.DataFrame:
    """
    Create one directed edge per row where DSLINKNO != -1.
    Edge direction: LINKNO → DSLINKNO (upstream → downstream).
    """
    rows = []

    for _, reach in channels.iterrows():
        src_linkno = int(reach["LINKNO"])
        dst_linkno = int(reach["DSLINKNO"])

        # Skip outlet reaches (no downstream)
        if dst_linkno == OUTLET_SENTINEL:
            continue

        # Both ends must be in the mapping to be included
        if src_linkno not in linkno_map:
            continue
        if dst_linkno not in linkno_map:
            continue

        rows.append({
            "LINKNO":        src_linkno,
            "DSLINKNO":      dst_linkno,
            "src_subbasin":  linkno_map[src_linkno],
            "dst_subbasin":  linkno_map[dst_linkno],
        })

    edges = pd.DataFrame(rows)

    log.info(
        f"Directed edges constructed: {len(edges)} edges "
        f"({len(channels) - len(edges)} outlet/unmatched reaches excluded)"
    )

    # Self-loop check
    self_loops = (edges["src_subbasin"] == edges["dst_subbasin"]).sum()
    if self_loops:
        log.warning(
            f"{self_loops} self-loops detected (LINKNO maps to same subbasin "
            f"as DSLINKNO). These will be removed."
        )
        edges = edges[edges["src_subbasin"] != edges["dst_subbasin"]]

    return edges


def _attach_edge_attributes(
    edges:    pd.DataFrame,
    channels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge static edge attributes from Channels_KB.csv onto the edge list.
    """
    available = [c for c in EDGE_ATTR_COLS if c in channels.columns]

    if not available:
        log.warning("No edge attribute columns found in Channels_KB.csv.")
        return edges

    attr_df = channels[["LINKNO"] + available].copy()
    edges   = edges.merge(attr_df, on="LINKNO", how="left")

    # Fill missing attribute values with 0
    for col in available:
        null_count = edges[col].isna().sum()
        if null_count:
            log.warning(f"  Edge attribute '{col}': {null_count} NaN → filled with 0")
        edges[col] = edges[col].fillna(0)

    log.info(f"Edge attributes attached: {available}")
    return edges


def _map_to_node_indices(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add src_idx and dst_idx columns (0-based node indices for PyG).
    """
    id_to_idx = nodes.set_index("subbasin_id")["node_idx"].to_dict()

    edges["src_idx"] = edges["src_subbasin"].map(id_to_idx)
    edges["dst_idx"] = edges["dst_subbasin"].map(id_to_idx)

    unmapped = edges[["src_idx", "dst_idx"]].isna().any(axis=1).sum()
    if unmapped:
        log.warning(
            f"{unmapped} edges have unmapped node indices and will be dropped. "
            f"Check that subbasin_id in nodes.csv covers all LINKNO values."
        )

    edges = edges.dropna(subset=["src_idx", "dst_idx"]).copy()
    edges["src_idx"] = edges["src_idx"].astype(int)
    edges["dst_idx"] = edges["dst_idx"].astype(int)

    log.info(f"Node index mapping complete: {len(edges)} valid edges remaining")
    return edges


def _save_confluence_summary(channels: pd.DataFrame) -> None:
    """
    Save a summary of confluences (reaches with two upstream tributaries).
    Uses USLINKNO1 and USLINKNO2 to identify bifurcations/confluences.
    """
    if "USLINKNO1" not in channels.columns or "USLINKNO2" not in channels.columns:
        log.info("USLINKNO1/USLINKNO2 not found — skipping confluence summary.")
        return

    confluences = channels[
        (channels["USLINKNO1"] != OUTLET_SENTINEL) &
        (channels["USLINKNO2"] != OUTLET_SENTINEL)
    ][["LINKNO", "DSLINKNO", "USLINKNO1", "USLINKNO2"]].copy()

    out_path = GRAPH_DIR / "edge_topology.csv"
    confluences.to_csv(out_path, index=False)

    log.info(
        f"Confluence summary: {len(confluences)} reaches have 2 upstream "
        f"tributaries → {out_path}"
    )


def _validate_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    """
    Validate the constructed graph and write a stats report.
    Checks: DAG property, connectivity, degree distribution.
    """
    log.info("Validating graph topology...")

    G = nx.DiGraph()

    # Add all nodes
    for _, row in nodes.iterrows():
        G.add_node(int(row["node_idx"]))

    # Add all edges
    for _, row in edges.iterrows():
        G.add_edge(int(row["src_idx"]), int(row["dst_idx"]))

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    in_deg  = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]

    # DAG check (river networks should not have cycles)
    is_dag = nx.is_directed_acyclic_graph(G)
    if not is_dag:
        cycles = list(nx.simple_cycles(G))[:5]
        log.warning(
            f"Graph contains cycles (not a DAG). "
            f"First 5 cycles: {cycles}\n"
            f"River networks should be acyclic — check Channels_KB.csv."
        )
    else:
        log.info("Graph is a DAG (no cycles) ✓")

    # Connectivity
    n_components = nx.number_weakly_connected_components(G)
    if n_components > 1:
        log.warning(
            f"Graph has {n_components} weakly connected components. "
            f"Expected 1 for a single river basin."
        )
    else:
        log.info("Graph is weakly connected (single component) ✓")

    # Isolated nodes (no edges at all)
    isolated = list(nx.isolates(G))
    if isolated:
        log.warning(
            f"{len(isolated)} isolated nodes (no edges): {isolated[:10]}"
        )

    # Stats report
    stats = {
        "Nodes":                    n_nodes,
        "Edges":                    n_edges,
        "Is DAG":                   is_dag,
        "Connected components":     n_components,
        "Isolated nodes":           len(isolated),
        "Max in-degree":            max(in_deg)  if in_deg  else 0,
        "Max out-degree":           max(out_deg) if out_deg else 0,
        "Mean in-degree":           round(np.mean(in_deg),  3) if in_deg  else 0,
        "Mean out-degree":          round(np.mean(out_deg), 3) if out_deg else 0,
        "Nodes with in-degree 0":   sum(d == 0 for d in in_deg),
        "Nodes with out-degree 0":  sum(d == 0 for d in out_deg),
    }

    report_path = GRAPH_DIR / "graph_stats.txt"
    with open(report_path, "w") as f:
        f.write("Graph Validation Report\n")
        f.write("=" * 40 + "\n")
        for k, v in stats.items():
            f.write(f"{k:<30}: {v}\n")

    for k, v in stats.items():
        log.info(f"  {k:<30}: {v}")

    log.info(f"Graph stats saved → {report_path}")