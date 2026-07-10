"""
feature_engineering.py
=======================
Computes drought indices and assembles the final node feature matrix.

Drought Indices computed:
  SPI  (Standardized Precipitation Index)    — scales 1, 3, 6, 12
  SSI  (Standardized Streamflow Index)       — scales 1, 3, 6, 12
  SRI  (Standardized Runoff Index)           — scales 1, 3, 6, 12  ← TARGET
  GWI  (Groundwater Index)                   — scales 3, 6, 12

Final node feature vector (F = 11 dynamic + 4 static):
  Dynamic:  rainfall, runoff, soil_moisture, ET, groundwater,
            streamflow, temperature, reservoir_release, ENSO, SPI, SRI
  Static:   elevation, area, slope, drainage_density
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.utils.logger import get_logger

log = get_logger(__name__)

PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Drought index computation
# ---------------------------------------------------------------------------

def compute_spi(rainfall: pd.Series, scale: int = 3) -> pd.Series:
    """
    Standardized Precipitation Index using empirical normal approximation.
    rainfall: monthly series sorted by time for ONE sub-basin.
    """
    rolled = rainfall.rolling(window=scale, min_periods=scale).sum()
    mu  = rolled.mean()
    std = rolled.std()
    spi = (rolled - mu) / (std + 1e-8)
    return spi


def compute_sri(runoff: pd.Series, scale: int = 3) -> pd.Series:
    """Standardized Runoff Index — same formula as SPI but on runoff."""
    rolled = runoff.rolling(window=scale, min_periods=scale).sum()
    mu  = rolled.mean()
    std = rolled.std()
    sri = (rolled - mu) / (std + 1e-8)
    return sri


def compute_ssi(streamflow: pd.Series, scale: int = 3) -> pd.Series:
    """Standardized Streamflow Index."""
    return compute_sri(streamflow, scale)


def compute_gwi(groundwater: pd.Series, scale: int = 3) -> pd.Series:
    """Groundwater drought index."""
    rolled = groundwater.rolling(window=scale, min_periods=scale).mean()
    mu  = rolled.mean()
    std = rolled.std()
    return (rolled - mu) / (std + 1e-8)


def compute_all_indices(df: pd.DataFrame,
                        spi_scales:  list[int] = [1, 3, 6, 12],
                        sri_scales:  list[int] = [1, 3, 6, 12],
                        ssi_scales:  list[int] = [1, 3, 6, 12],
                        gwi_scales:  list[int] = [3, 6, 12]) -> pd.DataFrame:
    """
    Compute all drought indices for every sub-basin.
    df must have columns: subbasin_id, date, rainfall_mm (or runoff_mm etc.)
    """
    results = []

    for sb_id, grp in df.groupby("subbasin_id"):
        grp = grp.sort_values("date").copy()

        for s in spi_scales:
            if "rainfall_mm" in grp.columns:
                grp[f"spi_{s}"] = compute_spi(grp["rainfall_mm"], s).values

        for s in sri_scales:
            if "runoff_mm" in grp.columns:
                grp[f"sri_{s}"] = compute_sri(grp["runoff_mm"], s).values

        for s in ssi_scales:
            if "streamflow_m3s" in grp.columns:
                grp[f"ssi_{s}"] = compute_ssi(grp["streamflow_m3s"], s).values

        for s in gwi_scales:
            if "groundwater_mm" in grp.columns:
                grp[f"gwi_{s}"] = compute_gwi(grp["groundwater_mm"], s).values

        results.append(grp)

    df_out = pd.concat(results, ignore_index=True)
    log.info(f"Drought indices computed. Shape: {df_out.shape}")
    return df_out


def classify_drought(sri: pd.Series) -> pd.Series:
    """
    Classify SRI into drought severity categories.
      0 = Normal      (SRI >= -0.5)
      1 = Mild        (-1.0 <= SRI < -0.5)
      2 = Moderate    (-1.5 <= SRI < -1.0)
      3 = Severe      (SRI < -1.5)
    """
    cats = pd.cut(
        sri,
        bins=[-np.inf, -1.5, -1.0, -0.5, np.inf],
        labels=[3, 2, 1, 0],
        right=False,
    ).astype(float)
    return cats


def build_node_features(df: pd.DataFrame,
                        static_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the final node feature DataFrame.
    Dynamic features (11): rainfall, runoff, soil_moisture, ET, groundwater,
                           streamflow, temperature, reservoir_release, ENSO, spi_3, sri_3
    Static features (4):   elevation, area, slope, drainage_density

    static_df must have columns: subbasin_id, elevation, area_km2, slope, drainage_density
    """
    dynamic_cols = [
        "subbasin_id", "date",
        "rainfall_mm", "runoff_mm", "soil_moisture_mm", "et_mm",
        "groundwater_mm", "streamflow_m3s", "temperature_c",
        "release_mcm", "enso_index", "spi_3", "sri_3",
    ]
    existing = [c for c in dynamic_cols if c in df.columns]
    features = df[existing].copy()

    # Merge static attributes
    features = features.merge(
        static_df[["subbasin_id", "elevation_m", "area_km2",
                   "slope_pct", "drainage_density"]],
        on="subbasin_id", how="left"
    )

    out_path = PROCESSED / "node_features.csv"
    features.to_csv(out_path, index=False)
    log.info(f"Node features saved → {out_path}")
    return features
