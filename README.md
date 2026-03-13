# osiro1
# Castle-SeedFormer

## �T�v
�{�v���W�F�N�g�́A�_�Q�⊮���f�� **[SeedFormer](https://github.com/hrzhou2/seedformer.git)** ���x�[�X�ɁA���{�̏�s���z�i�����_�ЂȂǁj�̑�K�͓_�Q�f�[�^�����Ƀt�@�C���`���[�j���O���s�����J�X�^�����f���ł��B

�]���̎�@�ł͍�������A1���_�K�͂̋���ȓ_�Q�f�[�^�̏������\�ɂ��邽�߁A�f�[�^�̕���������w�K�v���Z�X�̍œK�����s���܂����B��X�́A����̃W�I���}�f�[�^���w�K�ɗp���A�X�}�[�g�t�H���ŎB�e�������݂��邨��̓_�Q�f�[�^��⊮�E�C�����邱�Ƃɐ������Ă��܂��B

## �f�� / ���s����
���ۂɌ�����������̓_�Q�f�[�^���C���������ʂł��B

<p align="center">
  <img src="picture/���_�O.png" width="45%" alt="���_�O">
  <img src="picture/���_��.png" width="45%" alt="���_��">
</p>
<p align="center">
  <b>Left: Input (Defective) | Right: Output (Restored)</b>
</p>
## ��ȋ@�\�ƕύX�_
�I���W�i����SeedFormer�ɑ΂��A�ȉ��̉��ǂ������Ă��܂��B

### 1. ��K�͓_�Q�ւ̑Ή�
* **�ۑ�**: �I���W�i�����f���� `partial` ���͂̏������2048�_�ł���A1���_�𒴂��邨��̃f�[�^�����̂܂܊w�K�����邱�Ƃ͕s�\�ł����B
* **������**: ��K�͓_�Q���ׂ��������i�`�����N���j���Ċw�K����p�C�v���C�����\�z���܂����B

### 2. �f�[�^�N�����W���O�Ɗw�K�̌�����
* �����̍ہA�_����2048�_�ɖ����Ȃ��`�����N�̓m�C�Y�ƌ��Ȃ��A�w�K�f�[�^���珜�O���鏈����ǉ����܂����B
* �e `partial` �f�[�^�ɂ����āA�ł��_�Q���x�������i�_���������j���_��D��I�ɍ̗p���郍�W�b�N���������A�w�K���������コ���܂����B

### 3. ��v�ȕύX�t�@�C��
* `train_pcn.py`: �w�K�v���Z�X�̍œK��
* `data_loader.py`: ��K�̓f�[�^�̕����ǂݍ��ݑΉ�
* `manager.py`: �w�K�Ǘ����W�b�N�̕ύX
* `run_castle_inference.py`: ���_���s�p�X�N���v�g�̒���

## ���\�z
* Python 3.x
* PyTorch
* Open3D
* **GCC/G++ 9** (Required for compiling custom CUDA ops)

## �g�p���@ (Workflow)

### 1. �f�[�^�Z�b�g�̍쐬 (Training)
�����̃f�[�^�i�����_�ЂȂǁj�Ŋw�K���s���ꍇ�̎菇�ł��B

1.  �w�K����������K�͓_�Q�f�[�^�iPLY�`���Ȃǁj��p�ӂ��܂��B
2.  **`�w�K�f�[�^���`�p.ipynb`** �����s���܂��B
    * ���̃m�[�g�u�b�N�́A���̓f�[�^�� `complete`�i���t�f�[�^�j�� `partial`�i5���_����̌����f�[�^�j�ɐ��`���A`train_pcn.py` �ň�����f�[�^�Z�b�g�`���ɕϊ����܂��B
3.  �w�K�����s���܂��B
    * **����**: �J�X�^���I�y���[�V�����̃R���p�C���ɂ� `gcc-9` / `g++-9` ���K�v�ł��B
    ```bash
    CC=gcc-9 CXX=g++-9 python3 train_pcn.py
    ```

### 2. ���_�̎��s (Inference)
�w�K�ς݃��f�����g���āA�����̂���f�[�^���C������菇�ł��B

1.  �⊮�������_�Q�t�@�C���� **`���_�f�[�^���`�p.ipynb`** �ɓ��͂��A�O�������s���܂��B
2.  `run_castle_inference.py` ���� `path` �ϐ����A���`�����f�[�^�̃p�X�ɕύX���܂��B
3.  �X�N���v�g�����s���܂��B
    ```bash
    CC=gcc-9 CXX=g++-9 python3 run_castle_inference.py
    ```

## �c�_�E����̓W�]
* **�⊮�͂Ɛ��x�̃g���[�h�I�t**:
    ���_���Ɍ������������ꍇ�A`train_pcn.py` �̃p�����[�^�𒲐����邱�ƂŁu�⊮�́v�����߂邱�Ƃ͉\�ł��B�������A�⊮�͂����߂�����ƑS�̂��ڂ₯���i���������ꂽ�j�o�͂ɂȂ�X��������A�f�B�e�[���Ƃ̃g���[�h�I�t���������Ă��܂��B
* **�r�W���A���̉��P**:
    ���_���ʂ̌����ځi�e�N�X�`���▧�x�Ȃǁj��ǋ�����ꍇ�́A`run_castle_inference.py` �̌㏈���A���S���Y��������ɉ��ǂ���K�v������܂��B

## �Q�l�����E�N���W�b�g
* Original Model: [SeedFormer](https://github.com/hrzhou2/seedformer.git)

## ����
* **���� �s���N (Shuntaro Kogami)**
* ��B��w �H�w�� �d�C���H�w�� (Kyushu University, Department of Electrical and Information Engineering)
# Castle-SeedFormer

## Overview
This project is a custom point cloud completion model fine-tuned for large-scale Japanese castle architecture (such as castles and shrines), based on **[SeedFormer](https://github.com/hrzhou2/seedformer.git)**.

Processing massive point cloud data (approx. 100 million points) is difficult for conventional methods. To address this, we optimized the data splitting process and the training pipeline. We successfully trained the model using castle diorama data and performed inference/restoration on real-world castle point clouds scanned with smartphones.

## Demo / Results
![Result Comparison]([Insert Image Path Here])

## Key Features & Modifications
We have made the following improvements to the original SeedFormer:

### 1. Support for Large-Scale Point Clouds
* **Challenge**: The original model had an input limit of approximately 2048 points per `partial` input, making it impossible to directly train on castle data with over 100 million points.
* **Solution**: We implemented a pipeline to split (chunk) large-scale point clouds into smaller segments for training.

### 2. Data Cleansing & Training Efficiency
* **Noise Filtering**: Added a process to exclude chunks with fewer than 2048 points during splitting to eliminate noise.
* **Viewpoint Optimization**: Implemented logic to prioritize and select the viewpoint with the highest point density for each `partial` dataset, improving training efficiency.

### 3. Key Modified Files
* `train_pcn.py`: Optimized the training process.
* `data_loader.py`: Adapted for loading split large-scale data.
* `manager.py`: Modified training management logic.
* `run_castle_inference.py`: Adjusted for inference execution.

## Internal Processing Flow

This section explains what actually happens inside `seedformer-master/codes/` during training and inference — written for readers without a deep machine-learning background.

---

### Training Flow

The goal of training is to teach the AI model a general rule: *"given a point cloud with missing parts, predict what the complete shape should look like."*

```
┌─────────────────────────────────────────────────────┐
│  Input Data (prepared in advance)                   │
│                                                     │
│  complete/*.pcd  ─── Ground truth (full 3D scan)    │
│  partial/*.pcd   ─── Defective scan (missing parts) │
│                      * 5 viewpoints per model       │
└───────────────────┬─────────────────────────────────┘
                    │  train_pcn.py  (entry point)
                    │  Reads config, sets up data pipeline
                    ▼
┌─────────────────────────────────────────────────────┐
│  Data Loader  (utils/data_loaders.py)               │
│  Shuffles and feeds data in batches of 8 pairs      │
│  to the GPU at a time                               │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  SeedFormer Model  (model.py)   — 3-stage pipeline  │
│                                                     │
│  Stage 1 — FeatureExtractor                         │
│    2,048 input points → compressed 1,024-dim        │
│    feature vector  (captures overall shape)         │
│                                                     │
│  Stage 2 — SeedGenerator                           │
│    Uses feature vector to generate 256 coarse       │
│    "seed" points distributed over the full shape    │
│                                                     │
│  Stage 3 — UpTransformer  (×1 → ×4 → ×4)           │
│    Progressively refines seeds into a dense         │
│    16,384-point completed point cloud               │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Loss Calculation  (utils/loss_utils.py)            │
│  Measures how far the predicted shape is from the   │
│  ground truth using Chamfer Distance (CD) —         │
│  a score where lower = more accurate.               │
│  Evaluated at each of the 3 stages.                 │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Optimizer  (manager.py)                            │
│  Adam optimizer with gradual warmup LR schedule.    │
│  Runs for 400 epochs. After each epoch,             │
│  validates on held-out data and saves the best      │
│  performing weights → ckpt-best.pth                 │
└─────────────────────────────────────────────────────┘
```

**Key parameters** (set in `train_pcn.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `N_INPUT_POINTS` | 2,048 | Points fed into the model per sample |
| `UPSAMPLE_FACTORS` | [1, 4, 4] | Output grows to 16,384 points (2048×1×4×4) |
| `BATCH_SIZE` | 8 | Samples processed simultaneously |
| `N_EPOCHS` | 400 | Total training passes over the dataset |
| `LEARNING_RATE` | 0.001 | Initial step size for weight updates |
| `WARMUP_EPOCHS` | 20 | Ramp-up period before full learning rate |

---

### Inference Flow

The goal of inference is to take a real, large-scale castle scan (tens of millions of points) and fill in the missing or sparse regions using the trained model.

Because the model accepts exactly 2,048 points at a time, the large scan is divided into small overlapping cube-shaped patches. Each patch is repaired independently, and all results are merged at the end.

```
┌─────────────────────────────────────────────────────┐
│  Input                                              │
│  Large castle PLY file  (tens of millions of pts)   │
└───────────────────┬─────────────────────────────────┘
                    │  run_castle_inference.py
                    ▼
┌─────────────────────────────────────────────────────┐
│  Patch Center Generation                            │
│  Voxel downsample the scan at stride=0.02 m         │
│  → produces a grid of patch centers covering        │
│    the entire structure                             │
└───────────────────┬─────────────────────────────────┘
                    │  For each patch center (loop)
                    ▼
┌─────────────────────────────────────────────────────┐
│  Per-Patch Processing                               │
│                                                     │
│  (a) Extract all points within a 0.25 m cube        │
│      around the center                              │
│  (b) If fewer than 256 points → skip (empty region) │
│  (c) Sample or upsample to exactly 2,048 points     │
│  (d) Normalize: shift centroid to origin, scale     │
│      so the patch fits in a unit sphere             │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Model Inference  (trained ckpt-best.pth)           │
│  Input: 2,048 normalized points                     │
│  Output: 16,384 completed points                    │
│  Inverse-transform back to original coordinates     │
└───────────────────┬─────────────────────────────────┘
                    │  After all patches are processed
                    ▼
┌─────────────────────────────────────────────────────┐
│  Post-Processing                                    │
│                                                     │
│  1. SOR Filter — removes statistical outliers       │
│  2. Voxel Downsample (0.0005 m) — unifies density   │
│  3. Color Transfer — assigns colors to new points   │
│     by weighted average of K=3 nearest original pts │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Output                                             │
│  Original scan + generated points merged into       │
│  a single colored PLY file                          │
└─────────────────────────────────────────────────────┘
```

**Key parameters** (set at the top of `run_castle_inference.py`):

| Parameter | Value | Meaning |
|---|---|---|
| `PATCH_SIZE` | 0.25 m | Side length of each cube patch |
| `CENTER_STRIDE` | 0.02 m | Spacing between patch centers |
| `MIN_KEEP` | 256 pts | Minimum points required to process a patch |
| `FINAL_VOXEL_SIZE` | 0.0005 m | Output point spacing after downsampling |

---

### File Roles at a Glance

| File | Role |
|---|---|
| `train_pcn.py` | Entry point for training — loads config and starts the pipeline |
| `manager.py` | Manages the training loop, validation, and checkpoint saving |
| `model.py` | Defines the SeedFormer neural network architecture |
| `utils/data_loaders.py` | Reads and batches the training dataset |
| `utils/loss_utils.py` | Computes Chamfer Distance loss at each stage |
| `run_castle_inference.py` | Entry point for inference on real castle point clouds |

---

## Requirements
* Python 3.x
* PyTorch
* Open3D
* **GCC/G++ 9** (Required for compiling custom CUDA operations)

## Usage Workflow

### 1. Dataset Creation (Training)
Follow these steps to train the model with your own data (castles, shrines, etc.):

1.  Prepare the large-scale point cloud data (e.g., PLY format).
2.  Run the notebook **`�w�K�f�[�^���`�p.ipynb` (Training_Data_Preprocessing.ipynb)**.
    * This notebook formats the input data into `complete` (ground truth) and `partial` (defective data from 5 viewpoints) sets suitable for `train_pcn.py`.
3.  Run the training script.
    * **Note**: `gcc-9` and `g++-9` are required to compile the custom operations.
    ```bash
    CC=gcc-9 CXX=g++-9 python3 train_pcn.py
    ```

### 2. Inference
Follow these steps to restore defective point clouds using the trained model:

1.  Input the target point cloud file into **`���_�f�[�^���`�p.ipynb` (Inference_Data_Preprocessing.ipynb)** for preprocessing.
2.  Update the `path` variable in `run_castle_inference.py` to point to the formatted data.
3.  Run the inference script:
    ```bash
    python3 run_castle_inference.py
    ```

## Issues & Future Outlook
* **Trade-off between Completion Strength and Precision**:
    When significant parts are missing during inference, modifying parameters in `train_pcn.py` can improve the model's completion capability. However, stronger completion often results in blurrier (over-smoothed) outputs, creating a trade-off with detail retention.
* **Visual Improvements**:
    To achieve better visual quality in the inference results (textures, density), further modifications to the post-processing algorithms in `run_castle_inference.py` are required.

## References & Credits
* Original Model: [SeedFormer](https://github.com/hrzhou2/seedformer.git)

## Author
* **Shuntaro Kogami**
* Department of Electrical and Information Engineering, Kyushu University