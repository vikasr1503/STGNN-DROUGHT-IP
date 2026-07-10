"""
build_nodes.py
==============
Builds the node table for the Krishna River Basin sub-basin graph.

Each node = one SWAT sub-basin (130 nodes).

Input:
    data/raw/subbasin.csv       — Subbasin.csv from advisor
                                  Columns: PolygonId, Area, Subbasin
    data/raw/dem_attributes.csv — optional DEM-derived attributes
                                  Columns: subbasin_id, elevation_m,
                                           slope_pct, drainage_density,
                                           centroid_lat, centroid_lon

Output:
    data/graph/nodes.csv        — one row per sub-basin node
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log      = get_logger(__name__)
GRAPH_DIR = Path("data/graph")
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

# Expected columns in Subbasin.csv
REQUIRED_COLS = {"PolygonId", "Area", "Subbasin"}

# DEM attribute columns (filled with NaN if not provided)
DEM_COLS = [
    "elevation_m",
    "slope_pct",
    "drainage_density",
    "centroid_lat",
    "centroid_lon",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_node_table(
    subbasin_csv: str = "data/raw/subbasin.csv",
    dem_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build and validate the node attribute table.

    Parameters
    ----------
    subbasin_csv : path to Subbasin.csv (PolygonId, Area, Subbasin)
    dem_csv      : optional path to DEM-derived attributes per sub-basin

    Returns
    -------
    DataFrame with one row per sub-basin node, sorted by subbasin_id,
    with 0-based node_idx for PyTorch Geometric.
    """
    log.info("=" * 55)
    log.info("Building node table")
    log.info("=" * 55)

    # ── Load ────────────────────────────────────────────────────
    df = _load_subbasin_csv(subbasin_csv)

    # ── Validate ────────────────────────────────────────────────
    _validate_subbasin(df)

    # ── Transform ───────────────────────────────────────────────
    df = df.rename(columns={"Subbasin": "subbasin_id", "Area": "area_m2"})
    df["area_km2"] = (df["area_m2"] / 1e6).round(6)

    df = df.sort_values("subbasin_id").reset_index(drop=True)
    df["node_idx"] = df.index          # 0-based index for PyG

    # ── DEM attributes ──────────────────────────────────────────
    if dem_csv and Path(dem_csv).exists():
        df = _merge_dem(df, dem_csv)
    else:
        _add_placeholder_dem(df, dem_csv)

    # ── Save ────────────────────────────────────────────────────
    out_path = GRAPH_DIR / "nodes.csv"
    df.to_csv(out_path, index=False)

    log.info(f"Node table saved: {len(df)} nodes → {out_path}")
    log.info(f"Columns: {list(df.columns)}")
    _log_summary(df)

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_subbasin_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Subbasin CSV not found: {p}\n"
            f"Place Subbasin.csv (from advisor) at: {p}"
        )
    df = pd.read_csv(p)
    log.info(f"Loaded {p.name}: {len(df)} rows, columns={list(df.columns)}")
    return df


def _validate_subbasin(df: pd.DataFrame) -> None:
    """Run all validation checks on raw Subbasin.csv."""

    # Required columns
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Subbasin.csv is missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )

    # No duplicate Subbasin IDs
    dupes = df["Subbasin"].duplicated()
    if dupes.any():
        dup_ids = df.loc[dupes, "Subbasin"].tolist()
        raise ValueError(
            f"Duplicate Subbasin IDs found: {dup_ids}\n"
            f"Each sub-basin must have a unique ID."
        )

    # No missing Subbasin IDs
    if df["Subbasin"].isna().any():
        raise ValueError("Subbasin column contains NaN values.")

    # Area must be positive
    invalid_area = df["Area"] <= 0
    if invalid_area.any():
        bad = df.loc[invalid_area, "Subbasin"].tolist()
        raise ValueError(f"Non-positive area found for sub-basins: {bad}")

    # Check for gaps in Subbasin IDs (warn only, not error)
    ids   = sorted(df["Subbasin"].tolist())
    expected = list(range(min(ids), max(ids) + 1))
    gaps  = sorted(set(expected) - set(ids))
    if gaps:
        log.warning(f"Gaps in Subbasin ID sequence: {gaps}")
    else:
        log.info(f"Subbasin IDs are contiguous: {min(ids)} → {max(ids)}")

    log.info(
        f"Validation passed: {len(df)} sub-basins, "
        f"IDs {df['Subbasin'].min()} → {df['Subbasin'].max()}"
    )


def _merge_dem(df: pd.DataFrame, dem_csv: str) -> pd.DataFrame:
    dem = pd.read_csv(dem_csv)

    if "subbasin_id" not in dem.columns:
        raise ValueError(
            f"DEM CSV must have a 'subbasin_id' column. "
            f"Found: {list(dem.columns)}"
        )

    # Check all sub-basins are present in DEM file
    missing = set(df["subbasin_id"]) - set(dem["subbasin_id"])
    if missing:
        log.warning(
            f"{len(missing)} sub-basins have no DEM attributes "
            f"(will be NaN): {sorted(missing)[:10]}..."
        )

    df = df.merge(dem, on="subbasin_id", how="left")
    found_cols = [c for c in DEM_COLS if c in df.columns]
    log.info(f"Merged DEM attributes: {found_cols}")

    # Add any missing DEM columns as NaN
    for col in DEM_COLS:
        if col not in df.columns:
            df[col] = np.nan
            log.warning(f"DEM column '{col}' not found — set to NaN.")

    return df


def _add_placeholder_dem(df: pd.DataFrame, dem_csv: Optional[str]) -> None:
    for col in DEM_COLS:
        if col not in df.columns:
            df[col] = np.nan

    if dem_csv:
        log.warning(
            f"DEM CSV not found: {dem_csv}\n"
            f"DEM attributes set to NaN — fill from SWAT outputs before training."
        )
    else:
        log.warning(
            "No DEM CSV provided. "
            "DEM attributes (elevation, slope, etc.) set to NaN.\n"
            "Provide dem_csv= to merge real values."
        )


def _log_summary(df: pd.DataFrame) -> None:
    log.info(f"  Sub-basins     : {len(df)}")
    log.info(f"  node_idx range : 0 → {df['node_idx'].max()}")
    log.info(f"  Area (km²)     : min={df['area_km2'].min():.2f}  "
             f"max={df['area_km2'].max():.2f}  "
             f"mean={df['area_km2'].mean():.2f}")
    if df["centroid_lat"].notna().any():
        log.info(f"  Lat range      : {df['centroid_lat'].min():.3f} → "
                 f"{df['centroid_lat'].max():.3f}")
        log.info(f"  Lon range      : {df['centroid_lon'].min():.3f} → "
                 f"{df['centroid_lon'].max():.3f}")
    nan_dem = df[DEM_COLS].isna().all(axis=1).sum()
    if nan_dem:
        log.warning(
            f"  {nan_dem} sub-basin(s) have no DEM attributes (NaN). "
            "Provide DEM CSV before training."
        )