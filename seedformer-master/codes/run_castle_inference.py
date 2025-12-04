import torch
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'codes'))
import time
from importlib import import_module
from collections import OrderedDict

# --- 設定パラメータ ---
MODEL_PATH = '/home/limu/seedformer-master/codes/shapenet55-20251120T032419Z-1-001/shapenet55/ShapeNet-55/ckpt-best.pth' # ★パス確認
INPUT_PLY = '/home/limu/seedformer-master/kesson2.ply'             # 元の87万点のファイル
OUTPUT_PLY = 'kumamoto_overlap_repaired_SOR_cleaned2.ply'
DEVICE = torch.device('cuda:0')

N_INPUT_POINTS = 2048                   # モデルの入力サイズ
VOXEL_SIZE = 0.05                        # ★スライドさせる間隔 (小さいほど密にオーバーラップする)
# お城のスケールによりますが、0.2 ~ 1.0 くらいで調整してください

# --- メイン処理 ---
def run_inference():
    # 1. モデルのロード (学習時設定)
    print("Initializing model...")
    Model = import_module('model')
    model = Model.__dict__['seedformer_dim128'](
        up_factors=[1, 4, 4],
        num_p0=512
    )
    model = model.to(DEVICE)
    
    # 2. 重みロード
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights not found.")
        return
    
    print(f"Loading weights...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    original_state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in original_state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    # 3. 元データの読み込み
    print(f"Reading {INPUT_PLY}...")
    pcd_full = o3d.io.read_point_cloud(INPUT_PLY)
    points_full = np.asarray(pcd_full.points)
    print(f"Original points: {points_full.shape[0]}")

    # 4. KDTreeの構築 (高速検索用)
    print("Building KDTree for overlapping search...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_full)

    # 5. 中心点の決定 (ボクセルダウンサンプリングで間引いて中心点を決める)
    # これにより、全体をまんべんなくカバーできる
    pcd_centers = pcd_full.voxel_down_sample(voxel_size=VOXEL_SIZE)
    centers = np.asarray(pcd_centers.points)
    n_patches = centers.shape[0]
    print(f"Generated {n_patches} patch centers (Voxel Size: {VOXEL_SIZE})")

    all_repaired_parts = []

    print("Starting Patch-based Inference...")
    # 各中心点についてループ
    for i in range(n_patches):
        center_point = centers[i]

        SEARCH_RADIUS = 1.5 # ★ 1.5メートル以内の点を集める
        [k, idx, _] = pcd_tree.search_radius_vector_3d(center_point, SEARCH_RADIUS)

# 点が多すぎる場合はランダムに2048点に減らす
        if k >= N_INPUT_POINTS:
            idx = np.random.choice(np.asarray(idx), N_INPUT_POINTS, replace=False)
        else:
            continue # 点が足りないスカスカの場所はスキップ
            
        # パッチの抽出
        patch_points = points_full[idx, :] # (2048, 3)

        # (B) 正規化 (Normalization) : これが超重要！
        # パッチを原点に持ってきて、サイズを1に収める
        centroid = np.mean(patch_points, axis=0)
        patch_centered = patch_points - centroid
        scale = np.max(np.sqrt(np.sum(patch_centered**2, axis=1)))
        patch_normalized = patch_centered / scale

        # (C) 推論
        tensor_in = torch.from_numpy(patch_normalized).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pcds_pred = model(tensor_in)
        
        # 結果取得 (8192点)
        pred_normalized = pcds_pred[-1].squeeze(0).cpu().numpy()

        # (D) 復元 (Denormalization) : 元の座標系に戻す
        # 正規化の逆計算を行う: (推論結果 * スケール) + 中心座標
        pred_restored = (pred_normalized * scale) + centroid
        
        all_repaired_parts.append(pred_restored)

        if (i+1) % 50 == 0:
            print(f"Processed {i+1}/{n_patches} patches...")

    # 6. 結合と保存
    print("Merging results...")
    if not all_repaired_parts:
        print("No patches processed.")
        return

    final_points = np.vstack(all_repaired_parts)
    
   # ... (前回のコードの続き。ループが終わって final_points ができたところから) ...

    print("Merging results...")
    if not all_repaired_parts:
        print("No patches processed.")
        return

    # AIが生成した全点群
    generated_points = np.vstack(all_repaired_parts)
    
    # --- ★ここからが修正箇所：ノイズ除去と結合 ---
    print("Filtering noise (keeping only points that fill holes)...")
    
    # 1. Open3Dの形式に変換
    pcd_gen = o3d.geometry.PointCloud()
    pcd_gen.points = o3d.utility.Vector3dVector(generated_points)
    
    # 2. 元の点群との距離を計算
    # (generated_points の各点が、original_points からどれくらい離れているか)
    dists = pcd_gen.compute_point_cloud_distance(pcd_full)
    dists = np.asarray(dists)
    
    # 3. しきい値設定 (メートル単位)
    # 元の点群から「この距離」以上離れている点だけを採用する
    # 小さすぎるとノイズが残り、大きすぎると穴埋めが消える。0.05 (5cm) ~ 0.1 (10cm) くらい推奨
    THRESHOLD = 0.05 
    
    # 4. フィルタリング (遠い点だけ残す)
    mask = dists > THRESHOLD
    points_filling_holes = generated_points[mask]
    
    print(f"Original AI points: {generated_points.shape[0]}")
    print(f"Points filling holes: {points_filling_holes.shape[0]} (Removed {generated_points.shape[0] - points_filling_holes.shape[0]} noise points)")

    # 5. 「元の綺麗な点群」+「穴埋め点群」を結合
    final_combined_points = np.vstack((points_full, points_filling_holes))
    
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(final_combined_points)
    
    # 色の扱い: 元の点群に色がある場合、新しい点にも色を塗るか、赤色などで目立たせる
    if pcd_full.has_colors():
        # 元の点群の色を取得
        colors_original = np.asarray(pcd_full.colors)
        # 新しい点群を「赤色」にする（どこが埋まったか分かりやすくするため）
        colors_new = np.tile(np.array([1.0, 0.0, 0.0]), (points_filling_holes.shape[0], 1))
        # 結合
        final_colors = np.vstack((colors_original, colors_new))
        out_pcd.colors = o3d.utility.Vector3dVector(final_colors)

    print("\nStarting SOR filtering...")
    
    # 考慮する近傍点の数 (nb_neighbors): 20 ~ 50 程度
    NB_NEIGHBORS = 30 
    # 標準偏差の閾値 (std_ratio): 1.0 ~ 3.0 程度
    STD_RATIO = 2.0 
    
    # フィルター適用
    # remove_statistical_outliers は、ノイズを除去したPCDと、残った点のインデックスを返す
    out_pcd_filtered, ind = out_pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO
    )
    
    # ノイズ除去後の点群 (インライア) だけを新しい out_pcd とし、色情報も引き継ぐ
    out_pcd.points = out_pcd_filtered.points
    if out_pcd_filtered.has_colors():
        out_pcd.colors = out_pcd_filtered.colors
        
    print(f"Total points before SOR: {final_combined_points.shape[0]}")
    print(f"Total points after SOR: {np.asarray(out_pcd.points).shape[0]}")

    o3d.io.write_point_cloud(OUTPUT_PLY, out_pcd)
    print(f"Saved cleaned & combined result to {OUTPUT_PLY}")
    print(f"Total points: {final_combined_points.shape[0]}")

if __name__ == '__main__':
    run_inference()
