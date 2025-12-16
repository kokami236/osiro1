import torch
import numpy as np
import open3d as o3d
import os
import sys

# 実行ディレクトリに codes フォルダがある前提
sys.path.append(os.path.join(os.getcwd(), "codes"))
from importlib import import_module
from collections import OrderedDict

# --- 設定パラメータ ---
MODEL_PATH = "/home/limu/seedformer-master/results/train_pcn_Log_2025_12_15_05_11_11/checkpoints/ckpt-best.pth"
INPUT_PLY = "/home/limu/seedformer-master/kesson2.ply"
OUTPUT_PLY = (
    "kumamoto_overlap_repaired_SOR_cleaned_downsampled.ply"  # 出力ファイル名変更
)
DEVICE = torch.device("cuda:0")

N_INPUT_POINTS = 2048  # モデルの入力サイズ
VOXEL_SIZE = 0.05  # パッチ抽出時の中心点間隔
FINAL_VOXEL_SIZE = 0.005  # ★追加: 最終的なダウンサンプリングの間隔


# --- メイン処理 ---
def run_inference():
    # 1. モデルのロード
    print("Initializing model...")
    Model = import_module("model")
    model = Model.__dict__["seedformer_dim128"](up_factors=[1, 4, 4], num_p0=512)
    model = model.to(DEVICE)

    # 2. 重みロード
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights not found at {MODEL_PATH}")
        return

    print(f"Loading weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    original_state_dict = checkpoint["model"]
    new_state_dict = OrderedDict()
    for k, v in original_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 3. 元データの読み込み
    print(f"Reading {INPUT_PLY}...")
    pcd_full = o3d.io.read_point_cloud(INPUT_PLY)
    points_full = np.asarray(pcd_full.points)
    print(f"Original points: {points_full.shape[0]}")

    # 4. KDTreeの構築
    print("Building KDTree for overlapping search...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)

    # 5. 中心点の決定
    pcd_centers = pcd_full.voxel_down_sample(voxel_size=VOXEL_SIZE)
    centers = np.asarray(pcd_centers.points)
    n_patches = centers.shape[0]
    print(f"Generated {n_patches} patch centers (Voxel Size: {VOXEL_SIZE})")

    all_repaired_parts = []

    print("Starting Patch-based Inference...")
    for i in range(n_patches):
        center_point = centers[i]
        SEARCH_RADIUS = 1.5
        [k, idx, _] = pcd_tree.search_radius_vector_3d(center_point, SEARCH_RADIUS)

        if k >= N_INPUT_POINTS:
            idx = np.random.choice(np.asarray(idx), N_INPUT_POINTS, replace=False)
        else:
            continue

        patch_points = points_full[idx, :]

        # 正規化
        centroid = np.mean(patch_points, axis=0)
        patch_centered = patch_points - centroid
        scale = np.max(np.sqrt(np.sum(patch_centered**2, axis=1)))
        patch_normalized = patch_centered / scale

        # 推論
        tensor_in = torch.from_numpy(patch_normalized).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pcds_pred = model(tensor_in)

        pred_normalized = pcds_pred[-1].squeeze(0).cpu().numpy()

        # 復元
        pred_restored = (pred_normalized * scale) + centroid
        all_repaired_parts.append(pred_restored)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{n_patches} patches...")

    # 6. 生成点群の結合
    print("Merging results...")
    if not all_repaired_parts:
        print("No patches processed.")
        return

    generated_points = np.vstack(all_repaired_parts)

    # --- ノイズ除去と結合 ---
    print("Filtering noise (keeping only points that fill holes)...")

    pcd_gen = o3d.geometry.PointCloud()
    pcd_gen.points = o3d.utility.Vector3dVector(generated_points)

    # 元の点群との距離計算
    dists = pcd_gen.compute_point_cloud_distance(pcd_full)
    dists = np.asarray(dists)

    THRESHOLD = 0.05  # 5cm以上離れた点のみ採用（穴埋め用）
    mask = dists > THRESHOLD
    points_filling_holes = generated_points[mask]

    print(f"Original AI points: {generated_points.shape[0]}")
    print(f"Points filling holes: {points_filling_holes.shape[0]}")

    # 結合
    final_combined_points = np.vstack((points_full, points_filling_holes))

    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(final_combined_points)

    # 色情報の処理
    if pcd_full.has_colors():
        colors_original = np.asarray(pcd_full.colors)
        colors_new = np.tile(
            np.array([1.0, 0.0, 0.0]), (points_filling_holes.shape[0], 1)
        )
        final_colors = np.vstack((colors_original, colors_new))
        out_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    # --- SOR フィルタリング ---
    print("\nStarting SOR filtering...")
    NB_NEIGHBORS = 30
    STD_RATIO = 0.1

    out_pcd_filtered, ind = out_pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO
    )

    # フィルタリング結果を適用
    out_pcd.points = out_pcd_filtered.points
    if out_pcd_filtered.has_colors():
        out_pcd.colors = out_pcd_filtered.colors

    print(f"Points before SOR: {final_combined_points.shape[0]}")
    print(f"Points after SOR: {np.asarray(out_pcd.points).shape[0]}")

    # --- ★追加箇所：最後のダウンサンプリング ---
    print(f"\nFinal Downsampling (Voxel Size: {FINAL_VOXEL_SIZE})...")

    # Voxel Downsampling 実行
    out_pcd_final = out_pcd.voxel_down_sample(voxel_size=FINAL_VOXEL_SIZE)

    print(f"Points after Downsampling: {np.asarray(out_pcd_final.points).shape[0]}")

    # 保存
    o3d.io.write_point_cloud(OUTPUT_PLY, out_pcd_final)
    print(f"Saved cleaned, merged & downsampled result to {OUTPUT_PLY}")


if __name__ == "__main__":
    run_inference()
