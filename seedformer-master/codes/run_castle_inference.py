import torch
import numpy as np
import open3d as o3d
import os, sys, math
from importlib import import_module
from collections import OrderedDict
# ★高速化のために追加
from scipy.spatial import cKDTree

# 実行ディレクトリに codes フォルダがある前提
sys.path.append(os.path.join(os.getcwd(), "codes"))

# --- 設定パラメータ ---
MODEL_PATH = "/home/limu/seedformer-master/results/train_pcn_Log_2026_01_18_23_14_21/checkpoints/ckpt-best.pth"
INPUT_PLY = "/home/limu/seedformer-master/codes/kakegawakesson8.ply"
OUTPUT_PLY = "final_kakegawa_colored_udogakusyuu2.ply"
DEVICE = torch.device("cuda:0")
N_INPUT_POINTS = 2048

# ★パッチサイズ
PATCH_SIZE = 0.25

# ★ストライド (0.02でも高速化すれば動くはずです)
CENTER_STRIDE = 0.02 

MIN_KEEP = 256
JITTER_SIGMA = 0.005
JITTER_CLIP = 0.02

# ★出力の解像度
FINAL_VOXEL_SIZE = 0.0005
SEARCH_MARGIN = 1.05

def _upsample_with_jitter(pts, n_points, sigma=0.005, clip=0.02):
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
    model = Model.dict["seedformer_dim128"](up_factors=[1, 4, 4], num_p0=512).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights not found at {MODEL_PATH}")
        return

    print("Loading weights...")
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
    
    if pcd_full.has_colors():
        colors_full = np.asarray(pcd_full.colors)
    else:
        print("Warning: Input PLY has no colors. Using default gray.")
        colors_full = np.tile(np.array([0.5, 0.5, 0.5]), (points_full.shape[0], 1))
        pcd_full.colors = o3d.utility.Vector3dVector(colors_full)
        
    print(f"Original points: {points_full.shape[0]}")

    # 3) KDTree (Open3D用: 近傍探索用)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)

    # 4) パッチ中心
    pcd_centers = pcd_full.voxel_down_sample(voxel_size=CENTER_STRIDE)
    centers = np.asarray(pcd_centers.points).astype(np.float32)
    n_patches = centers.shape[0]
    print(f"Generated {n_patches} patch centers (stride={CENTER_STRIDE})")

    # 5) 推論ループ
    search_radius = (PATCH_SIZE * math.sqrt(3) / 2.0) * SEARCH_MARGIN
    half = PATCH_SIZE / 2.0
    all_repaired_parts = []
    processed = 0

    print("Starting Inference...")
    for i, center_point in enumerate(centers):
        # (a) 近傍探索
        k, idx, _ = pcd_tree.search_radius_vector_3d(center_point, search_radius)
        if k <= 0:
            continue
        cand = points_full[np.asarray(idx)]

        # (b) Crop
        minb = center_point - np.array([half, half, half], dtype=np.float32)
        maxb = center_point + np.array([half, half, half], dtype=np.float32)
        mask = np.all((cand >= minb) & (cand < maxb), axis=1)
        patch_points = cand[mask]

        if patch_points.shape[0] < MIN_KEEP:
            continue

        # (c) Sampling
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

        # (d) Normalize & Inference
        centroid = np.mean(patch_points, axis=0, dtype=np.float32)
        patch_centered = patch_points - centroid
        scale = np.max(np.linalg.norm(patch_centered, axis=1)) + 1e-8
        patch_normalized = patch_centered / scale

        tensor_in = torch.from_numpy(patch_normalized).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pcds_pred = model(tensor_in)
            pred_normalized = pcds_pred[-1].squeeze(0).cpu().numpy().astype(np.float32)

        # (f) Restore
        pred_restored = (pred_normalized * scale) + centroid
        all_repaired_parts.append(pred_restored)
        processed += 1

        if (i + 1) % 100 == 0:
            print(f"center {i+1}/{n_patches} | processed={processed}")

    if not all_repaired_parts:
        print("No patches processed.")
        return

    generated_points = np.vstack(all_repaired_parts).astype(np.float32)
    print(f"Generated points total: {generated_points.shape[0]}")

    # --- 生成点群の処理 ---
    pcd_gen = o3d.geometry.PointCloud()
    pcd_gen.points = o3d.utility.Vector3dVector(generated_points)

    print("Applying SOR filter...")
    pcd_gen, _ = pcd_gen.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    print(f"Applying Voxel Downsample ({FINAL_VOXEL_SIZE})...")
    pcd_gen_ds = pcd_gen.voxel_down_sample(voxel_size=FINAL_VOXEL_SIZE)
    
    # --------------------------------------------------------------------------
    # ★高速化修正点：色の補間処理（Scipy cKDTreeによる一括計算）
    # --------------------------------------------------------------------------
    print("🎨 Transferring colors using Fast Weighted KNN (Scipy)...")
    gen_pts = np.asarray(pcd_gen_ds.points)
    
    # K近傍
    K_NEIGHBORS = 3
    
    # ScipyのKDTreeを構築（高速検索用）
    print(f"Building KDTree for {points_full.shape[0]} original points...")
    tree = cKDTree(points_full)
    
    # 一括検索 (query returns distances and indices)
    # n_jobs=-1 でCPU全コア使用
    print(f"Querying {K_NEIGHBORS} nearest neighbors for {gen_pts.shape[0]} generated points...")
    dists, indices = tree.query(gen_pts, k=K_NEIGHBORS, workers=-1) 
    
    # 距離のゼロ除算回避
    dists = np.maximum(dists, 1e-8)
    
    # 重み計算 (IDW: Inverse Distance Weighting)
    weights = 1.0 / dists
    
    # 重みの正規化 (各行の和で割る)
    # keepdims=True で (N, 1) の形を維持してブロードキャスト可能にする
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_norm = weights / weights_sum
    
    # インデックスに対応する色を取得 (N, K, 3)
    neighbor_colors = colors_full[indices]
    
    # 重み付き平均計算
    # (N, K, 1) * (N, K, 3) -> (N, K, 3) -> sum -> (N, 3)
    weighted_colors = np.sum(neighbor_colors * weights_norm[:, :, np.newaxis], axis=1)
    
    # 色を割り当て
    pcd_gen_ds.colors = o3d.utility.Vector3dVector(weighted_colors)
    print("Color transfer complete.")

    # --------------------------------------------------------------------------
    # マージして保存
    # --------------------------------------------------------------------------
    print("🔗 Merging original and generated point clouds...")
    final_pcd = pcd_full + pcd_gen_ds
    o3d.io.write_point_cloud(OUTPUT_PLY, final_pcd)
    print(f"✅ Saved final result to: {OUTPUT_PLY}")

if __name__ == "__main__":
    run_inference()