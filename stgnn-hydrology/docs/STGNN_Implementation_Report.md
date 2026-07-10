# STGNN Implementation Report
## Multi-Source Hydrological Drought Prediction — Krishna River Basin

### 1. Overview
This report documents the implementation of a Physics-Informed Dynamic Directed Spatio-Temporal Graph Neural Network for hydrological drought prediction across 130 sub-basins of the Krishna River Basin.

### 2. Data Pipeline
- Raw data ingested from India-WRIS, SWAT model outputs, groundwater, reservoir sources
- Atmospheric variables processed via `atmos_master_pipeline.py`
- All sources aligned to monthly resolution and 130 sub-basin spatial units

### 3. Graph Construction
- 130 nodes (SWAT sub-basins, Krishna River Basin)
- Directed edges following river network topology (upstream→downstream)
- Dynamic edge weights incorporating runoff, reservoir release, seasonality, SPI

### 4. Drought Indices
SPI, SRI, SSI, GWI computed at scales 1, 3, 6, 12 months.
Primary prediction target: SRI-3, SRI-6, SRI-12.

### 5. Model Progression
| Phase | Architecture | Status |
|-------|-------------|--------|
| 1 | GCN + GRU | Implemented |
| 2 | GAT + GRU | Implemented |
| 3 | Graph WaveNet | Implemented |

### 6. Training Configuration
See `configs/train.yaml` for full hyperparameter details.

### 7. Results
(To be filled after experiments complete)

### 8. Conclusions
(To be filled after experiments complete)
