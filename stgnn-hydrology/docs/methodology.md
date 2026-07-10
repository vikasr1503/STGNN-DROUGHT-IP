# Methodology

## Problem Statement
Predict hydrological drought (SRI-3, SRI-6, SRI-12) across 130 sub-basins of the Krishna River Basin using a Spatio-Temporal Graph Neural Network that captures both spatial river connectivity and temporal patterns.

## Data Sources
| Source | Variables | Frequency |
|--------|-----------|-----------|
| India-WRIS (NWIC) | Temperature, Pressure, Solar Radiation, RH, Wind | Monthly avg |
| SWAT Model | Runoff, Soil Moisture, ET, Streamflow, Groundwater | Monthly |
| River Discharge | Observed streamflow | Monthly |
| Groundwater | Well levels | Monthly |
| Reservoir | Storage, Release | Monthly |

## Graph Construction
- **Nodes**: 130 SWAT sub-basins (Krishna River Basin)
- **Edges**: Directed upstream→downstream river connectivity
- **Dynamic Edge Weights**: `W(A,B,t) = f(Runoff, FlowAcc, Season, ReservoirRelease, SPI, TravelTime)`

## Node Features (F = 11 dynamic + 4 static)
Dynamic: Rainfall, Runoff, Soil Moisture, ET, Groundwater, Streamflow, Temperature, Reservoir Release, ENSO, SPI-3, SRI-3
Static: Elevation, Area, Slope, Drainage Density

## Drought Index Computation
- **SPI**: Standardized Precipitation Index (scales 1,3,6,12)
- **SRI**: Standardized Runoff Index (scales 1,3,6,12) — primary target
- **SSI**: Standardized Streamflow Index (scales 1,3,6,12)
- **GWI**: Groundwater Index (scales 3,6,12)

Classification thresholds (SRI):
- Normal: SRI ≥ -0.5
- Mild: -1.0 ≤ SRI < -0.5
- Moderate: -1.5 ≤ SRI < -1.0
- Severe: SRI < -1.5

## Model Architecture (Phase 1: GCN+GRU)
```
Input [batch, 12, 130, 11]
  → GCN × 2 (per timestep)
  → GRU × 2 (across 12 months)
  → Linear head
  → Output [batch, 130, 3]
```

## Training Protocol
- Period: 1970–2025 (monthly)
- Train: up to 2015-12
- Validation: 2016-01 to 2019-12
- Test: 2020-01 onwards
- Lookback: 12 months
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: 0.7×MSE + 0.3×MAE (masked for NaN targets)
- Early stopping: patience=15 epochs
