# Experimental Design

## Experiment A — Discharge Only (F=1)
Feature: streamflow only.
Baseline: does basic streamflow predict SRI?

## Experiment B — Discharge + Atmosphere (F=6)
Features: streamflow, temperature, rainfall, humidity, ET, pressure.
Research question: does atmospheric data improve drought prediction?

## Experiment C — Discharge + Atmosphere + Groundwater (F=7)
Add groundwater level.

## Experiment D — Full System (F=11+)
All features including reservoir release, ENSO, SPI.

## Baseline Models
| Model | Type |
|-------|------|
| Persistence | predict t-1 as t |
| Random Forest | per-node tabular |
| XGBoost | per-node tabular |
| LSTM | temporal only, no graph |
| GRU | temporal only, no graph |
| GCN+GRU | Phase 1 STGNN |
| GAT+GRU | Phase 2 STGNN |
| Graph WaveNet | Phase 3 STGNN |

## Evaluation Metrics
Regression: RMSE, MAE, R², NSE (Nash-Sutcliffe), KGE (Kling-Gupta)
Classification: Accuracy, Precision, Recall, F1 (per drought class)
