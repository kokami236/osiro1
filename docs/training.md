# Training — Castle-SeedFormer

> This document covers the full training pipeline: architecture, loss, hyperparameters, and the custom modifications made for large-scale castle point clouds.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
  - [Stage 1 — Feature Extractor](#stage-1--feature-extractor)
  - [Stage 2 — Seed Generator](#stage-2--seed-generator)
  - [Stage 3 — UpTransformer](#stage-3--uptransformer)
- [Loss Function](#loss-function)
- [Training Configuration](#training-configuration)
- [Custom Modifications](#custom-modifications)
- [Training a New Model](#training-a-new-model)

---

## Overview

Castle-SeedFormer is a **point cloud completion model** fine-tuned for Japanese castle architecture.  
The task is: given a **partial** point cloud (missing regions from occlusion or limited scan coverage), predict the **complete** 3D shape.

The base architecture is [SeedFormer](https://github.com/hrzhou2/seedformer), extended with a custom pipeline to handle real-world castle scans that can exceed **100 million points** — far beyond what the original model was designed for.

```
Input:  2,048-point partial scan  (one small patch of the castle)
Output: 16,384-point completed shape
```

---

## Model Architecture

The model processes point clouds in **three stages**, progressively refining from a compressed global feature down to a dense, detailed output.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Castle-SeedFormer                            │
│                                                                     │
│  Partial Input (2,048 pts)                                          │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────┐                                                │
│  │FeatureExtractor │  ← PointNet++ + vTransformer (encoder)        │
│  │  2048→512→128→1 │    hierarchical downsampling + attention       │
│  └────────┬────────┘                                                │
│           │  global feature (1024-dim) + local patch (128 pts)     │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ SeedGenerator   │  ← UpTransformer + MLP residual blocks        │
│  │  128 → 256 pts  │    generates coarse "seed" skeleton            │
│  └────────┬────────┘                                                │
│           │  256 seed points                                        │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  UpTransformer  │  ← Attention-based upsampling ×2              │
│  │  256→1024→16384 │    ×4 each stage, guided by local geometry     │
│  └────────┬────────┘                                                │
│           │                                                         │
│  Completed Output (16,384 pts)                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

### Stage 1 — Feature Extractor

**Role:** Compress the 2,048-point input into a compact global feature vector, while preserving local geometry.

The encoder follows the **PointNet++ set abstraction** design — progressively grouping and downsampling points, similar to how a CNN builds a feature pyramid from pixels.

```
Input: (B, 3, 2048)
  │
  ├─ SA Module 1 ──► (B, 128, 512)   # 2048 → 512 points, 128-dim features
  │   └─ vTransformer                # self-attention over local k-NN neighbors
  │
  ├─ SA Module 2 ──► (B, 256, 128)   # 512 → 128 points, 256-dim features
  │   └─ vTransformer
  │
  └─ SA Module 3 ──► (B, 1024, 1)   # 128 → 1 global vector (1024-dim)

Output: global feature (1024-dim) + local patch features (128 points, 256-dim)
```

The **vTransformer** inserted at each level applies self-attention within local neighborhoods (`k=20` nearest neighbors), letting each point aggregate context from its surroundings before being passed to the next level.

---

### Stage 2 — Seed Generator

**Role:** Decode the global feature into 256 coarse "seed" points — a sparse skeleton that captures the overall shape.

The design is conceptually similar to the **decoder** in sequence-to-sequence Transformers: a compact representation (global feature) is expanded into a structured output (seed points) via cross-attention and residual MLP blocks.

```
Global feature (1024-dim)  +  local patch (128 pts, 256-dim)
  │
  ├─ UpTransformer ──► 256 pts (upsample the 128-pt patch ×2)
  ├─ MLP_Res (128-dim)
  ├─ MLP_Res (128-dim)
  └─ MLP_Res (seed_dim) ──► Conv1d ──► seed XYZ coordinates (256 pts)
```

Output: 256 seed points + their feature embeddings (passed to Stage 3)

---

### Stage 3 — UpTransformer

**Role:** Upsample the 256 seed points to 16,384 dense points through two ×4 upsampling steps.

Each upsampling step uses the **UpTransformer**, which queries the local geometry via `k`-NN attention and duplicates each point `up_factor` times with position-aware offsets. This is similar in spirit to **ViT's patch-expansion decoder** — using attention to spread spatial information rather than simple interpolation.

```
Seed points (256 pts)
  │
  ├─ UpTransformer (×4) ──► 1,024 pts
  └─ UpTransformer (×4) ──► 16,384 pts   ← final output

Each UpTransformer step:
  key   = Conv1d(features)
  query = Conv1d(features)
  value = Conv1d(features)
  attn  = softmax(query · key^T / √dim)
  out   = attn · value  +  positional MLP(Δxyz)
  → replicated up_factor times per point
```

**Upsampling configuration** (`UPSAMPLE_FACTORS = [1, 4, 4]`):

| Stage | Points | Factor |
|-------|--------|--------|
| Seeds | 256 | — |
| UpTransformer 1 | 1,024 | ×4 |
| UpTransformer 2 | 16,384 | ×4 |

---

## Loss Function

Training minimizes **Chamfer Distance (CD)** at each stage simultaneously.  
CD measures the average nearest-neighbor distance between two point sets — lower is better.

```
L_total = L_cdc + L_cd1 + L_cd2 + L_cd3 + L_partial
```

| Term | Compared | Role |
|------|----------|------|
| `L_cdc` | Input partial ↔ Ground truth | Keeps output consistent with the input scan |
| `L_cd1` | Stage 2 output (256 pts) ↔ GT | Supervises coarse seed generation |
| `L_cd2` | Stage 3 mid (1,024 pts) ↔ GT | Supervises intermediate upsampling |
| `L_cd3` | Final output (16,384 pts) ↔ GT | **Main loss** — final output quality |
| `L_partial` | Final output ↔ Input partial | Prevents hallucinating geometry that contradicts the input |

CD is computed as the **square root** version (`sqrt=True`) during training, which is more sensitive to large outliers than the squared version.

```python
CD(S1, S2) = (1/|S1|) Σ_{x∈S1} min_{y∈S2} ||x−y|| + (1/|S2|) Σ_{y∈S2} min_{x∈S1} ||x−y||
```

---

## Training Configuration

All hyperparameters are defined in `PCNConfig()` inside `train_pcn.py`.

### Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | `1e-3` |
| β₁, β₂ | `0.9`, `0.999` |
| Weight decay | `0` |

### Learning Rate Schedule

A **gradual warmup** followed by **step decay** — the same pattern used in the original Transformer ("warm up, then decay"):

```
Epochs 1–20  : linear warmup  (lr: 0 → 1e-3)
Epochs 21+   : StepLR decay   (÷10 every 150 epochs)

Resulting LR milestones: ~epoch 170, ~320
```

### Other Settings

| Parameter | Value |
|-----------|-------|
| Epochs | 400 |
| Batch size | 8 |
| Input points | 2,048 |
| Output points | 16,384 |
| Upsample factors | [1, 4, 4] |
| Num workers | 0 |

### Checkpoint

The best model (by `cd3` on validation set) is saved automatically:

```
results/train_pcn_Log_<timestamp>/checkpoints/ckpt-best.pth
```

---

## Custom Modifications

The following changes were made on top of the original SeedFormer to handle large-scale castle point clouds.

### 1. Patch-based Training Pipeline

**Problem:** The original model expects inputs of exactly 2,048 points. Castle scans contain 10–100 million points — direct training is impossible.

**Solution:** A preprocessing pipeline (`学習データ整形用.ipynb`) that:
1. Splits the full castle scan into overlapping cubic patches (0.25 m³)
2. Samples each patch to exactly 2,048 points
3. Exports each patch as an independent `partial/complete` training pair in `.pcd` format

### 2. Sparse Patch Filtering

Patches with fewer than **2,048 points** are discarded during preprocessing.  
These correspond to boundary regions or empty space and would introduce noisy, low-information training samples.

### 3. Densest-Viewpoint Selection

For each model, 5 viewpoints are simulated (virtual cameras at different angles).  
Rather than training on all 5, **only the densest viewpoint** (highest point count) is selected per model.  
This ensures each training sample has sufficient geometric coverage to be meaningful.

### 4. Patch-based Inference

At inference time, `run_castle_inference.py` applies the same patch logic to the target castle:

```
Full castle PLY (millions of pts)
  │
  ├─ Voxel downsample (0.02 m) → patch centers
  │
  └─ For each center:
       extract 0.25 m³ cube → sample to 2,048 pts → model → 16,384 pts
       │
       └─ After all patches: SOR filter → voxel downsample → color transfer → merge
```

---

## Training a New Model

```bash
# 1. Prepare dataset
#    Run 学習データ整形用.ipynb to generate partial/complete .pcd pairs
#    Then generate the index file:
python create_json.py

# 2. Update paths in train_pcn.py
#    DATASETS.CUSTOM.PARTIAL_POINTS_PATH
#    DATASETS.CUSTOM.COMPLETE_POINTS_PATH
#    DATASETS.CUSTOM.CATEGORY_FILE_PATH

# 3. Run training
CC=gcc-9 CXX=g++-9 python3 seedformer-master/codes/train_pcn.py

# 4. Monitor training logs
tail -f results/train_pcn_Log_<timestamp>/training.txt
```

Training progress is logged per iteration:

```
<n_itr>  <cd_pc>  <cd_p1>  <cd_p2>  <cd_p3>  <partial_matching>
```

All five CD values should decrease over time. `cd_p3` is the most important metric — it directly measures final output quality.
