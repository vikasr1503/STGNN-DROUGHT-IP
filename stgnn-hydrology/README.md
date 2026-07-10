# Spatio-Temporal Graph Neural Network for Hydrological Drought Prediction
## Krishna River Basin, India — BTP Project

---

## Overview

This repository implements a **Physics-Informed Dynamic Directed Spatio-Temporal Graph Neural Network (STGNN)** for predicting hydrological drought in the Krishna River Basin (130 sub-basins) using multi-source hydro-meteorological data.

**Target variable:** Standardized Runoff Index (SRI-3, SRI-6, SRI-12)

**Data sources:**
- River Discharge (India-WRIS / NWIC)
- Atmospheric Variables: Temperature, Pressure, Solar Radiation, Humidity, Wind (India-WRIS)
- Groundwater Levels
- Reservoir Storage & Release
- SWAT hydrological model outputs

---

## Project Structure

```
stgnn-hydrology/
├── configs/          YAML configs for model, graph, training
├── data/             raw → interim → processed → tensors
├── docs/             methodology, architecture, experiments
├── notebooks/        EDA and analysis notebooks
├── scripts/          train.py, evaluate.py, predict.py
├── src/
│   ├── preprocessing/   data loading, cleaning, feature engineering
│   ├── graph/           node/edge construction, adjacency matrix
│   ├── datasets/        tensor builder, sequence generator, DataLoader
│   ├── models/          GCN+GRU, GAT+GRU, Graph WaveNet
│   ├── training/        trainer, losses, callbacks
│   ├── evaluation/      metrics (RMSE, NSE, KGE, F1)
│   └── visualization/   graph plots, prediction plots, attention maps
├── tests/            unit tests
└── results/          figures, tables, comparison
```

---

## Quickstart

### 1. Setup environment
```bash
conda env create -f environment.yml
conda activate stgnn-drought
```

### 2. Place raw data
```
data/raw/
  subbasin.csv            ← provided by advisor (130 sub-basins)
  discharge/              ← monthly CSV files per station
  atmosphere/             ← outputs of atmos_master_pipeline.py
  swat_outputs/           ← SWAT monthly simulation CSV
  groundwater/            ← groundwater level CSVs
  reservoir/              ← reservoir storage/release CSVs
```

### 3. Train
```bash
python scripts/train.py --model gcn_gru
python scripts/train.py --model gat_gru --run_name experiment_gat
```

### 4. Evaluate
```bash
python scripts/evaluate.py --run_name gcn_gru
```

### 5. Predict
```bash
python scripts/predict.py --run_name gcn_gru --date 2023-06
```

---

## Model Architecture

```
Input [batch, lookback=12, N=130, F=11]
        ↓
   GCN × 2 layers     (per timestep, spatial aggregation)
        ↓
   GRU × 2 layers     (across lookback, temporal memory)
        ↓
   Linear head
        ↓
Output [batch, N=130, 3]   (SRI-3, SRI-6, SRI-12)
```

---

## Results (placeholder — update after experiments)

| Model     | SRI-3 RMSE | SRI-3 NSE | SRI-3 KGE |
|-----------|-----------|-----------|-----------|
| GCN+GRU   | —         | —         | —         |
| GAT+GRU   | —         | —         | —         |
| LSTM (baseline) | —   | —         | —         |

---

## Citation

> Vikas Reddy. "Multi-Source Hydrological Drought Prediction Using Spatio-Temporal Graph Neural Networks." BTP Report, 2026.
