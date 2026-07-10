# Model Architecture

## Phase 1: GCN + GRU (Baseline)
```
Input X [B, T=12, N=130, F=11]
│
├─ For each t in [0..11]:
│    x_t [B, N, F]
│    └→ GCNConv → BatchNorm → ReLU → Dropout    [B, N, 64]
│    └→ GCNConv → BatchNorm → ReLU → Dropout    [B, N, 64]
│
├─ Stack → [B, N, T, 64] → reshape [B*N, T, 64]
│
├─ GRU(hidden=128, layers=2)
│    h_n[-1] → [B*N, 128] → reshape [B, N, 128]
│
└─ Linear(128→64) → ReLU → Dropout → Linear(64→3)
   Output [B, N, 3]  (SRI-3, SRI-6, SRI-12)
```

## Phase 2: GAT + GRU
Replace GCNConv with GATv2Conv (4 attention heads).
Attention coefficients reveal which upstream sub-basins
contribute most to drought propagation.

## Phase 3: Graph WaveNet
- Adaptive adjacency matrix (learned node embeddings)
- Dilated causal convolutions (replace GRU)
- Skip connections across dilation levels
- Parallelisable — faster than RNN-based models

## Edge Weight Computation (Dynamic)
```
W(A→B, t) = w1·Runoff_A(t)
           + w2·FlowAccumulation_AB
           + w3·sin(2π·month/12)
           + w4·ReservoirRelease(t)
           + w5·SPI_A(t)
           + w6·(1/TravelTime_AB)

weights: [0.25, 0.15, 0.10, 0.20, 0.15, 0.15]
```
