"""
preprocess.py
=============
Cleans and aligns all data sources to a common:
  - Temporal resolution: monthly
  - Spatial resolution:  sub-basin level (130 nodes)
  - Period: 1970-01 to 2025-12

Output: data/interim/merged_monthly.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from src.utils.logger import get_logger

log = get_logger(__name__)

INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)


def align_to_monthly_subbasin(df: pd.DataFrame,
                               subbasin_col: str,
                               date_col: str,
                               value_cols: list[str],
                               all_subbasins: list,
                               date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Reindex a DataFrame to cover all (subbasin, month) combinations.
    Missing values are left as NaN for imputation later.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()

    idx = pd.MultiIndex.from_product(
        [all_subbasins, date_range],
        names=[subbasin_col, date_col]
    )
    full = pd.DataFrame(index=idx).reset_index()
    merged = full.merge(df[[subbasin_col, date_col] + value_cols],
                        on=[subbasin_col, date_col], how="left")
    return merged


def impute_missing(df: pd.DataFrame,
                   value_cols: list[str],
                   group_col: str = "subbasin_id") -> pd.DataFrame:
    """
    Impute missing values per sub-basin:
      1. Linear interpolation along time axis
      2. Forward/backward fill for edge months
    """
    df = df.copy()
    df = df.sort_values([group_col, "date"])

    for col in value_cols:
        if col not in df.columns:
            continue
        df[col] = (
            df.groupby(group_col)[col]
              .transform(lambda s: s.interpolate("linear").ffill().bfill())
        )
    return df


def merge_all_sources(discharge: pd.DataFrame,
                      swat: pd.DataFrame,
                      groundwater: pd.DataFrame,
                      reservoir: pd.DataFrame,
                      subbasins: list,
                      date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Merge all data sources into a single tidy DataFrame indexed by
    (subbasin_id, date).
    """
    # Start with SWAT outputs (most complete spatial coverage)
    base = align_to_monthly_subbasin(
        swat, "subbasin_id", "date",
        ["runoff_mm", "soil_moisture_mm", "et_mm",
         "groundwater_mm", "streamflow_m3s", "water_yield_mm"],
        subbasins, date_range
    )

    # Merge observed discharge
    if not discharge.empty:
        base = base.merge(
            discharge[["subbasin_id", "date", "discharge_m3s"]],
            on=["subbasin_id", "date"], how="left"
        )

    # Merge groundwater
    if not groundwater.empty:
        base = base.merge(
            groundwater[["subbasin_id", "date", "groundwater_m"]],
            on=["subbasin_id", "date"], how="left"
        )

    # Merge reservoir release (map reservoir to subbasin via lookup)
    if not reservoir.empty:
        res_monthly = (
            reservoir.groupby(["subbasin_id", "date"])
                     ["release_mcm"].sum().reset_index()
        )
        base = base.merge(res_monthly, on=["subbasin_id", "date"], how="left")

    log.info(f"Merged dataset: {base.shape[0]:,} rows × {base.shape[1]} cols")
    return base


def scale_features(df: pd.DataFrame,
                   feature_cols: list[str],
                   save_scaler: bool = True) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Z-score normalise feature columns.
    Saves the fitted scaler to data/interim/scaler.pkl for inference time.
    """
    scaler = StandardScaler()
    df = df.copy()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))

    if save_scaler:
        joblib.dump(scaler, INTERIM / "scaler.pkl")
        log.info("Scaler saved to data/interim/scaler.pkl")

    return df, scaler


def run_preprocessing(discharge, swat, groundwater, reservoir,
                      subbasin_ids: list,
                      start: str = "1970-01",
                      end: str   = "2025-12") -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns cleaned, merged, imputed DataFrame.
    Saves to data/interim/merged_monthly.csv.
    """
    date_range = pd.date_range(start=start, end=end, freq="MS")

    merged = merge_all_sources(
        discharge, swat, groundwater, reservoir,
        subbasin_ids, date_range
    )

    feature_cols = [
        "runoff_mm", "soil_moisture_mm", "et_mm",
        "groundwater_mm", "streamflow_m3s", "water_yield_mm",
        "discharge_m3s", "groundwater_m", "release_mcm"
    ]
    existing = [c for c in feature_cols if c in merged.columns]
    merged = impute_missing(merged, existing)

    out_path = INTERIM / "merged_monthly.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"Saved merged monthly data → {out_path}")

    return merged
