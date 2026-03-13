# Castle-SeedFormer

> 🇯🇵 [日本語版はこちら](#japanese) ｜ 🇬🇧 [English version here](#english)

---

<a name="japanese"></a>
# 🇯🇵 日本語版

## 概要

本プロジェクトは、点群補完モデル **[SeedFormer](https://github.com/hrzhou2/seedformer.git)**
をベースに、日本の城郭建築（お城や神社など）の大規模点群データ向けに
ファインチューニングを行ったカスタムモデルです。

従来の手法では困難だった、1億点規模の巨大な点群データの処理を可能にするため、
データの分割処理や学習プロセスの最適化を行いました。
我々は、お城のジオラマデータを学習に用い、スマートフォンで撮影した実在する
お城の点群データを補完・修復することに成功しています。

## デモ / 実行結果

実際に欠損したお城の点群データを修復した結果です。

<p align="center">
  <img src="picture/推論前.png" width="45%" alt="推論前">
  <img src="picture/推論後.png" width="45%" alt="推論後">
</p>
<p align="center">
  <b>Left: Input (Defective) | Right: Output (Restored)</b>
</p>

## リポジトリ構成とファイルの役割

```
osiro1/
├── seedformer-master/codes/    ← 【学習・推論のコア】すべての処理はここで動く
│   ├── train_pcn.py            ← 学習の実行スクリプト
│   ├── manager.py              ← 学習ループ・検証・保存管理
│   ├── model.py                ← SeedFormerのニューラルネットワーク定義
│   ├── run_castle_inference.py ← お城点群への推論実行スクリプト
│   └── utils/                  ← データ読み込み・損失関数など補助ツール
│
├── 学習データ整形用.ipynb       ← 【前処理】生の点群データを学習形式に変換
├── 推論データ整形用.ipynb       ← 【前処理】推論前の点群の前処理
└── （その他のルートファイル）   ← 上記2つのノートブックで使う補助スクリプト
```

> ルートにあるPythonファイルや `.ipynb` は、`seedformer-master/codes/` に
> データを渡すための「前処理専用」です。学習・推論の本体は `codes/` の中です。

## 内部処理フロー

### 学習フロー

AIモデルに「欠損した点群から完全な形状を予測するルール」を学ばせるプロセスです。

```
[入力データ]
  complete/*.pcd  ─ 正解データ（完全な3Dスキャン）
  partial/*.pcd   ─ 欠損データ（5視点からの不完全スキャン）
        ↓  train_pcn.py が設定を読み込みパイプラインを起動
[データローダー]  utils/data_loaders.py
  バッチ8件ずつシャッフルしてGPUへ転送
        ↓
[SeedFormerモデル]  model.py  ── 3段階で補完
  Stage1 FeatureExtractor  : 2,048点 → 形状の特徴量1,024次元に圧縮
  Stage2 SeedGenerator     : 特徴量から256点の粗い「種点」を生成
  Stage3 UpTransformer     : 種点を ×1→×4→×4 で拡張 → 16,384点
        ↓
[損失計算]  utils/loss_utils.py
  Chamfer距離（予測と正解のズレ）を3段階で計算 → 数値が小さいほど高精度
        ↓
[最適化]  manager.py
  Adamオプティマイザ + ウォームアップ付き学習率スケジューラ
  400エポック繰り返す → 最良モデルを ckpt-best.pth に保存
```

### 推論フロー

実際のお城の大規模点群（数千万点）に対して、欠損部分を補完するプロセスです。
モデルは一度に2,048点しか処理できないため、点群を小さなブロック（パッチ）に分割して処理します。

```
[入力]
  実際のお城の点群データ（PLYファイル・数千万点）
        ↓  run_castle_inference.py
[パッチ中心の生成]
  0.02m 間隔のボクセルグリッドでパッチ中心を設定
  → 建物全体をカバーする格子点群を生成
        ↓  各パッチ中心に対してループ処理
[パッチ単位の処理]
  (a) 中心から 0.25m の立方体内の点を抽出
  (b) 256点未満なら空領域としてスキップ
  (c) ちょうど 2,048点 にサンプリング
  (d) 重心を原点・スケールを正規化
        ↓
[モデル推論]  ckpt-best.pth（学習済み重み）
  入力: 2,048点  →  出力: 16,384点（補完済み）
  元の座標・スケールに逆変換
        ↓  全パッチ処理後
[後処理]
  1. SORフィルタ   ── 外れ値（ノイズ）を除去
  2. ボクセル間引き ── 点の密度を統一（0.0005m 解像度）
  3. 色転写       ── K=3近傍点の距離加重平均で色を付与
        ↓
[出力]
  元の点群 + 生成点群をマージした colored PLYファイル
```

## 主な機能と変更点

オリジナルのSeedFormerに対し、以下の改良を加えています。

### 1. 大規模点群への対応
* **課題**: オリジナルモデルは `partial` 入力の上限が約2048点であり、1億点を超えるお城のデータをそのまま学習させることは不可能でした。
* **解決策**: 大規模点群を細かく分割（チャンク化）して学習するパイプラインを構築しました。

### 2. データクレンジングと学習の効率化
* 分割の際、点数が2048点に満たないチャンクはノイズと見なし、学習データから除外する処理を追加しました。
* 各 `partial` データにおいて、最も点群密度が高い（点数が多い）視点を優先的に採用するロジックを実装し、学習効率を向上させました。

### 3. 主要な変更ファイル
* `train_pcn.py`: 学習プロセスの最適化
* `data_loader.py`: 大規模データの分割読み込み対応
* `manager.py`: 学習管理ロジックの変更
* `run_castle_inference.py`: 推論実行用スクリプトの調整

## 環境構築
* Python 3.x
* PyTorch
* Open3D
* **GCC/G++ 9**（カスタムCUDA演算のコンパイルに必要）

## 使用方法

### 1. データセットの作成・学習
1. 学習させたい大規模点群データ（PLY形式など）を用意します。
2. **`学習データ整形用.ipynb`** を実行します。
   * `complete`（教師データ）と `partial`（5視点からの欠損データ）に整形し、学習形式に変換します。
3. 学習を実行します。（`gcc-9` / `g++-9` が必要）
   ```bash
   CC=gcc-9 CXX=g++-9 python3 train_pcn.py
   ```

### 2. 推論の実行
1. 補完したい点群ファイルを **`推論データ整形用.ipynb`** で前処理します。
2. `run_castle_inference.py` 内のパス変数を更新します。
3. スクリプトを実行します。
   ```bash
   CC=gcc-9 CXX=g++-9 python3 run_castle_inference.py
   ```

## 議論・今後の展望
* **補完力と精度のトレードオフ**: 推論時に欠損が著しい場合、`train_pcn.py` のパラメータを調整することで「補完力」を高めることは可能です。しかし、補完力を強めすぎると全体がぼやけた（平滑化された）出力になる傾向があり、ディテールとのトレードオフが発生しています。
* **ビジュアルの改善**: 推論結果の見た目（テクスチャや密度など）を追求する場合は、`run_castle_inference.py` の後処理アルゴリズムをさらに改良する必要があります。

## 参考文献・クレジット
* Original Model: [SeedFormer](https://github.com/hrzhou2/seedformer.git)

## 著者
* **鴻上 峻太朗 (Kokami Shuntaro)**
* 九州大学 工学部 電気情報工学科

---

<a name="english"></a>
# 🇬🇧 English Version

## Overview

This project is a custom point cloud completion model fine-tuned for large-scale Japanese castle
architecture (such as castles and shrines), based on **[SeedFormer](https://github.com/hrzhou2/seedformer.git)**.

Processing massive point cloud data (approx. 100 million points) is difficult for conventional
methods. To address this, we optimized the data splitting process and the training pipeline.
We successfully trained the model using castle diorama data and performed inference/restoration
on real-world castle point clouds scanned with smartphones.

## Demo / Results

<p align="center">
  <img src="picture/推論前.png" width="45%" alt="Before">
  <img src="picture/推論後.png" width="45%" alt="After">
</p>
<p align="center">
  <b>Left: Input (Defective) | Right: Output (Restored)</b>
</p>

## Repository Structure

```
osiro1/
├── seedformer-master/codes/    ← [CORE] All training & inference runs here
│   ├── train_pcn.py            ← Training entry point
│   ├── manager.py              ← Training loop, validation, checkpoint management
│   ├── model.py                ← SeedFormer neural network architecture
│   ├── run_castle_inference.py ← Inference script for castle point clouds
│   └── utils/                  ← Data loaders, loss functions, helpers
│
├── 学習データ整形用.ipynb       ← [PREPROCESSING] Converts raw scan → training format
├── 推論データ整形用.ipynb       ← [PREPROCESSING] Prepares data before inference
└── (other root files)          ← Helper scripts used by the notebooks above
```

> Files at the root level are **preprocessing-only**. The core training and inference
> logic lives entirely inside `seedformer-master/codes/`.

## Internal Processing Flow

### Training Flow

The goal of training is to teach the AI model: *"given a point cloud with missing parts,
predict what the complete shape should look like."*

```
[Input Data]
  complete/*.pcd  ─ Ground truth (full 3D scan)
  partial/*.pcd   ─ Defective scan (5 viewpoints, missing regions)
        ↓  train_pcn.py  loads config, starts pipeline
[Data Loader]  utils/data_loaders.py
  Shuffles & feeds batches of 8 pairs to GPU
        ↓
[SeedFormer Model]  model.py  — 3-stage pipeline
  Stage 1 FeatureExtractor : 2,048 pts → 1,024-dim feature vector
  Stage 2 SeedGenerator    : generates 256 coarse "seed" points
  Stage 3 UpTransformer    : refines seeds ×1→×4→×4 → 16,384 pts
        ↓
[Loss]  utils/loss_utils.py
  Chamfer Distance (lower = more accurate) computed at each stage
        ↓
[Optimizer]  manager.py
  Adam + gradual warmup LR scheduler
  Runs 400 epochs → saves best model as ckpt-best.pth
```

### Inference Flow

Takes a large real-world castle scan (tens of millions of points) and fills missing regions.
The model processes one small cube-shaped patch at a time and merges all results.

```
[Input]  Large castle PLY file (tens of millions of points)
        ↓  run_castle_inference.py
[Patch Grid]
  Voxel downsample at 0.02 m → grid of patch centers covering the structure
        ↓  Loop over each center
[Per-Patch Processing]
  (a) Extract points within 0.25 m cube around center
  (b) Skip if fewer than 256 points (empty region)
  (c) Sample or upsample to exactly 2,048 points
  (d) Normalize: shift centroid to origin, scale to unit sphere
        ↓
[Model Inference]  ckpt-best.pth
  Input: 2,048 pts  →  Output: 16,384 completed pts
  Inverse-transform back to original coordinates
        ↓  After all patches
[Post-Processing]
  1. SOR Filter        — remove statistical outliers
  2. Voxel Downsample  — unify density at 0.0005 m
  3. Color Transfer    — weighted KNN color from original scan
        ↓
[Output]  Original scan + generated points → single colored PLY file
```

## Key Features & Modifications

### 1. Support for Large-Scale Point Clouds
* **Challenge**: The original model had an input limit of ~2,048 points per `partial` input,
  making it impossible to directly train on castle data with over 100 million points.
* **Solution**: Implemented a pipeline to split (chunk) large-scale point clouds into smaller
  segments for training.

### 2. Data Cleansing & Training Efficiency
* **Noise Filtering**: Chunks with fewer than 2,048 points are excluded from training data.
* **Viewpoint Optimization**: Logic to prioritize the viewpoint with the highest point density
  per `partial` dataset, improving training efficiency.

### 3. Key Modified Files

| File | Role |
|---|---|
| `train_pcn.py` | Training entry point — loads config, starts the pipeline |
| `manager.py` | Manages training loop, validation, and checkpoint saving |
| `model.py` | SeedFormer neural network architecture |
| `utils/data_loaders.py` | Reads and batches the training dataset |
| `utils/loss_utils.py` | Computes Chamfer Distance loss at each stage |
| `run_castle_inference.py` | Inference on real castle point clouds |

## Requirements
* Python 3.x
* PyTorch
* Open3D
* **GCC/G++ 9** (Required for compiling custom CUDA operations)

## Usage

### 1. Training
1. Prepare large-scale point cloud data (e.g., PLY format).
2. Run **`学習データ整形用.ipynb`** (Training Data Preprocessing notebook).
   * Converts raw data into `complete` (ground truth) and `partial` (5-viewpoint defective) sets.
3. Run training (`gcc-9` / `g++-9` required):
   ```bash
   CC=gcc-9 CXX=g++-9 python3 train_pcn.py
   ```

### 2. Inference
1. Run **`推論データ整形用.ipynb`** (Inference Data Preprocessing notebook) on target PLY.
2. Update the path variable in `run_castle_inference.py`.
3. Run inference:
   ```bash
   CC=gcc-9 CXX=g++-9 python3 run_castle_inference.py
   ```

## Issues & Future Outlook
* **Trade-off — Completion vs. Detail**: Stronger completion settings produce blurrier outputs.
* **Visual Quality**: Further post-processing improvements needed for texture and density.

## References & Credits
* Original Model: [SeedFormer](https://github.com/hrzhou2/seedformer.git)

## Author
* **Kokami Shuntaro** (鴻上 峻太朗)
* Department of Electrical and Information Engineering, Kyushu University
