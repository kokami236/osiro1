import torch
import numpy as np
import open3d as o3d
import os, sys, math
from importlib import import_module
from collections import OrderedDict

# 実行ディレクトリに codes フォルダがある前提
sys.path.append(os.path.join(os.getcwd(), "codes"))

# --- 設定パラメータ ---
MODEL_PATH = "/home/limu/seedformer-master/results/train_pcn_Log_2025_12_18_09_02_55/checkpoints/ckpt-best.pth"
INPUT_PLY  = "/home/limu/seedformer-master/codes/kakegawakesson2.ply"
OUTPUT_PLY = "suironkakegawa.ply"

DEVICE = torch.device("cuda:0")

N_INPUT_POINTS = 2048

# ★推論パッチを「25cm立方体」に揃える
PATCH_SIZE = 0.25          # 25cm cube
CENTER_STRIDE = 0.05       # パッチ中心の間隔（今のVOXEL_SIZE相当）

# 2048に満たないパッチをどうするか（荒れ対策）
MIN_KEEP = 256             # これ未満はスキップ（薄すぎる）
JITTER_SIGMA = 0.005       # 正規化空間での微小ノイズ（重複点の荒れ軽減）
JITTER_CLIP  = 0.02

# 出力後処理
FINAL_VOXEL_SIZE = 0.005   # 最終ダウンサンプル
FILL_THRESHOLD = 0.005      # ★変更: 5cm→2cm（補完点を捨てすぎないように）
SEARCH_MARGIN = 1.05       # 立方体を拾うための半径マージン


def _upsample_with_jitter(pts, n_points, sigma=0.005, clip=0.02):
    """ptsが少ない時に、重複の荒れを軽減するため微小jitter付きで水増し"""
    curr = pts.shape[0]
    if curr <= 0:
        return None
    idx = np.random.choice(curr, n_points, replace=True)
    out = pts[idx].copy()
    if sigma > 0:
        noise = sigma * np.random.randn(*out.shape)
        out += np.clip(noise, -clip, clip).astype(np.float32)
    return out.astype(np.float32)


def run_inference():
    # 1) モデルロード
    print("Initializing model...")
    Model = import_module("model")
    model = Model.__dict__["seedformer_dim128"](up_factors=[1, 4, 4], num_p0=512).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights not found at {MODEL_PATH}")
        return

    print("Loading weights...")
    # ★変更: weights_only の保険（PyTorch差異で落ちないように）
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    original_state_dict = checkpoint["model"]
    new_state_dict = OrderedDict()
    for k, v in original_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 2) 入力点群読み込み
    print(f"Reading {INPUT_PLY}...")
    pcd_full = o3d.io.read_point_cloud(INPUT_PLY)
    points_full = np.asarray(pcd_full.points).astype(np.float32)
    print(f"Original points: {points_full.shape[0]}")

    # 3) KDTree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)

    # 4) パッチ中心（stride=0.05m 間隔）
    pcd_centers = pcd_full.voxel_down_sample(voxel_size=CENTER_STRIDE)
    centers = np.asarray(pcd_centers.points).astype(np.float32)
    n_patches = centers.shape[0]
    print(f"Generated {n_patches} patch centers (stride={CENTER_STRIDE})")

    # 5) 25cm立方体に必要な半径（半対角）＋マージン
    search_radius = (PATCH_SIZE * math.sqrt(3) / 2.0) * SEARCH_MARGIN
    half = PATCH_SIZE / 2.0

    all_repaired_parts = []
    processed = 0
    skipped_thin = 0
    skipped_empty = 0

    print("Starting 25cm-cube Patch-based Inference...")
    for i, center_point in enumerate(centers):
        # (a) 半径で候補を拾う（高速化用）
        k, idx, _ = pcd_tree.search_radius_vector_3d(center_point, search_radius)
        if k <= 0:
            skipped_empty += 1
            continue

        cand = points_full[np.asarray(idx)]

        # (b) ★AABBで25cm立方体に絞る（ここが最重要）
        minb = center_point - np.array([half, half, half], dtype=np.float32)
        maxb = center_point + np.array([half, half, half], dtype=np.float32)
        mask = np.all((cand >= minb) & (cand < maxb), axis=1)
        patch_points = cand[mask]

        if patch_points.shape[0] < MIN_KEEP:
            skipped_thin += 1
            continue

        # (c) 2048点に揃える（荒れを避ける方針）
        if patch_points.shape[0] >= N_INPUT_POINTS:
            sel = np.random.choice(patch_points.shape[0], N_INPUT_POINTS, replace=False)
            patch_points = patch_points[sel].astype(np.float32)
        else:
            patch_points = _upsample_with_jitter(
                patch_points.astype(np.float32),
                N_INPUT_POINTS,
                sigma=JITTER_SIGMA,
                clip=JITTER_CLIP
            )

        # (d) 学習時と同じ正規化（中心=平均, scale=max距離）
        centroid = np.mean(patch_points, axis=0, dtype=np.float32)
        patch_centered = patch_points - centroid
        scale = np.max(np.linalg.norm(patch_centered, axis=1)) + 1e-8
        patch_normalized = patch_centered / scale

        # (e) 推論
        tensor_in = torch.from_numpy(patch_normalized).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pcds_pred = model(tensor_in)

        pred_normalized = pcds_pred[-1].squeeze(0).cpu().numpy().astype(np.float32)

        # (f) 復元
        pred_restored = (pred_normalized * scale) + centroid
        all_repaired_parts.append(pred_restored)
        processed += 1

        if (i + 1) % 100 == 0:
            print(f"center {i+1}/{n_patches} | processed={processed} | thin_skip={skipped_thin}")

    print(f"Processed patches: {processed}")
    print(f"Skipped (too thin <{MIN_KEEP}): {skipped_thin}")
    print(f"Skipped (empty): {skipped_empty}")

    # 6) 結合
    # 6) 結合（生成点群）
    if not all_repaired_parts:
        print("No patches processed.")
        return
    
    generated_points = np.vstack(all_repaired_parts).astype(np.float32)
    print("Merging results...")
    print(f"Generated points total: {generated_points.shape[0]}")

# --- 生成点群だけを PointCloud 化 ---
    pcd_gen = o3d.geometry.PointCloud()
    pcd_gen.points = o3d.utility.Vector3dVector(generated_points)
    pcd_gen.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([1.0,0.0,0.0], dtype=np.float32),
                (len(pcd_gen.points),1))
    )


# --- SOR（生成点群だけに適用）---
    print("Starting SOR filtering (generated only)...")
    print("Before SOR:", np.asarray(pcd_gen.points).shape[0])
    pcd_gen_f, ind = pcd_gen.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)  # ←0.1は厳しすぎ
    print("After  SOR:", np.asarray(pcd_gen_f.points).shape[0])
    pcd_gen = pcd_gen_f

# --- Downsample（生成点群だけ）---
    print(f"Final Downsampling (Voxel Size: {FINAL_VOXEL_SIZE})...")
    print("Before Downsample:", np.asarray(pcd_gen.points).shape[0])
    pcd_gen_ds = pcd_gen.voxel_down_sample(voxel_size=FINAL_VOXEL_SIZE)
    print("After  Downsample:", np.asarray(pcd_gen_ds.points).shape[0])

# --- 全点を赤に（元点群は入れないので混ざらない）---
    n = np.asarray(pcd_gen_ds.points).shape[0]
    pcd_gen_ds.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    )
    o3d.io.write_point_cloud("generated_only_red2.ply", pcd_gen_ds)
    print("Saved: generated_only_red2.ply")



if __name__ == "__main__":
    run_inference()
