"""
load_data.py
============
Loads all hydro-meteorological data sources for the Krishna River Basin.

Expected data layout in data/raw/:
  discharge/        — monthly streamflow CSV per station
  atmosphere/       — atmospheric pipeline outputs (from atmos_master_pipeline.py)
  groundwater/      — groundwater level CSVs
  reservoir/        — reservoir storage CSVs
  swat_outputs/     — SWAT simulation outputs (runoff, soil moisture, ET, etc.)
  subbasin.csv      — 130 sub-basin attribute table (provided by advisor)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)

RAW = Path("data/raw")


def load_subbasin_table() -> pd.DataFrame:
    """
    Load the 130-subbasin attribute table.
    Columns: PolygonId, Area, Subbasin
    """
    path = RAW / "subbasin.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"Subbasin": "subbasin_id", "Area": "area_m2"})
    df["area_km2"] = df["area_m2"] / 1e6
    log.info(f"Loaded subbasin table: {len(df)} sub-basins.")
    return df


def load_discharge(freq: str = "M") -> pd.DataFrame:
    """
    Load and aggregate river discharge data to monthly frequency.
    Returns tidy DataFrame: [date, subbasin_id, discharge_m3s]
    freq: 'M' for monthly, 'D' for daily
    """
    path = RAW / "discharge"
    frames = []
    for f in sorted(path.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        frames.append(df)

    if not frames:
        log.warning("No discharge files found in data/raw/discharge/")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    if freq == "M":
        df = (
            df.groupby(["subbasin_id", pd.Grouper(key="date", freq="MS")])
              ["discharge_m3s"].mean().reset_index()
        )

    log.info(f"Loaded discharge: {df.shape[0]:,} rows, "
             f"{df['subbasin_id'].nunique()} stations.")
    return df


def load_atmosphere() -> dict[str, pd.DataFrame]:
    """
    Load the 5 atmospheric variable CSVs produced by atmos_master_pipeline.py.
    Returns dict keyed by variable name, each a tidy monthly DataFrame.
    """
    atmos_dir = RAW / "atmosphere"
    variables = {
        "temperature":          "Temperature",
        "atmospheric_pressure": "Atmospheric_Pressure",
        "solar_radiation":      "Solar_Radiation",
        "relative_humidity":    "Relative_Humidity",
        "wind_direction":       "Wind_Direction",
    }
    result = {}
    for key, slug in variables.items():
        candidates = list(atmos_dir.glob(f"{slug}_All_Stations_*_ma.csv"))
        if not candidates:
            log.warning(f"No monthly file found for {key}.")
            continue
        df = pd.read_csv(candidates[0])
        df["Data Time"] = pd.to_datetime(df["Data Time"], errors="coerce")
        df = df.rename(columns={
            "Station Code":          "station_code",
            "Data Time":             "date",
            "Monthly Avg Data Value":"value",
        })
        df = df[["station_code", "date", "value"]].dropna(subset=["date"])
        result[key] = df
        log.info(f"Loaded {key}: {len(df):,} rows.")
    return result


def load_swat_outputs() -> pd.DataFrame:
    """
    Load SWAT simulation outputs aggregated to monthly sub-basin level.
    Expected columns: date, subbasin_id, runoff_mm, soil_moisture_mm,
                      et_mm, groundwater_mm, streamflow_m3s, water_yield_mm
    """
    path = RAW / "swat_outputs" / "swat_monthly.csv"
    if not path.exists():
        log.warning(f"SWAT output file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    log.info(f"Loaded SWAT outputs: {df.shape[0]:,} rows.")
    return df


def load_groundwater() -> pd.DataFrame:
    """
    Load groundwater level data.
    Expected columns: date, subbasin_id, groundwater_m
    """
    path = RAW / "groundwater"
    frames = []
    for f in sorted(path.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        frames.append(df)
    if not frames:
        log.warning("No groundwater files found.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded groundwater: {df.shape[0]:,} rows.")
    return df


def load_reservoir() -> pd.DataFrame:
    """
    Load reservoir storage and release data.
    Expected columns: date, reservoir_id, storage_mcm, release_mcm
    """
    path = RAW / "reservoir"
    frames = []
    for f in sorted(path.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"])
        frames.append(df)
    if not frames:
        log.warning("No reservoir files found.")
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded reservoir: {df.shape[0]:,} rows.")
    return df
