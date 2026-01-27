import torch
import numpy as np
import open3d as o3d
import os, sys, math
from importlib import import_module
from collections import OrderedDict
from tqdm import tqdm
from scipy.spatial import cKDTree

sys.path.append(os.path.join(os.getcwd(), "codes"))

MODEL_PATH = "/home/limu/seedformer-master/results/train_pcn_Log_2026_01_18_23_14_21/checkpoints/ckpt-best.pth"
INPUT_PLY  = "/home/limu/seedformer-master/codes/kakegawakesson8.ply"
OUTPUT_PLY = "final_kakegawa_z_sliced.ply"
DEVICE = torch.device("cuda:0")

N_INPUT_POINTS = 2048

PATCH_SIZE = 0.25 
STRIDE     = 0.10   
MIN_KEEP = 50      
JITTER_SIGMA = 0.005
JITTER_CLIP  = 0.02

FINAL_VOXEL_SIZE = 0.0005

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
    print("Initializing model...")
    Model = import_module("model")
    model = Model.dict["seedformer_dim128"](up_factors=[1, 4, 4], num_p0=512).to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Weights not found at \{MODEL_PATH\}")
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

    print(f"Reading \{INPUT_PLY\}...")
    pcd_full = o3d.io.read_point_cloud(INPUT_PLY)
    points_full = np.asarray(pcd_full.points).astype(np.float32)
    
    if pcd_full.has_colors():
        colors_full = np.asarray(pcd_full.colors)
    else:
        print("Warning: Input PLY has no colors. Using default gray.")
        colors_full = np.tile(np.array([0.5, 0.5, 0.5]), (points_full.shape[0], 1))
        pcd_full.colors = o3d.utility.Vector3dVector(colors_full)
        
    print(f"Original points: \{points_full.shape[0]\}")

    mins, maxs = points_full.min(axis=0), points_full.max(axis=0)
    
    z_steps = np.arange(mins[2], maxs[2], STRIDE)
    total_steps = len(z_steps)
    
    print(f"Slice Config: Z-Range[\{mins[2]:.2f\} ~ \{maxs[2]:.2f\}], Steps=\{total_steps\}, PatchSize=\{PATCH_SIZE\}")

    all_repaired_parts = []
    processed_count = 0

    print("Starting Z-Slice Inference...")
    
    for z in tqdm(z_steps, desc="Processing Slices"):

        z_bottom = z
        z_top = z + PATCH_SIZE
        
        mask = (points_full[:, 2] >= z_bottom) & (points_full[:, 2] < z_top)
        patch_points = points_full[mask]

        if patch_points.shape[0] < MIN_KEEP:
            continue

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

        centroid = np.mean(patch_points, axis=0, dtype=np.float32)
        patch_centered = patch_points - centroid
        
        scale = np.max(np.linalg.norm(patch_centered, axis=1)) + 1e-8
        patch_normalized = patch_centered / scale

        tensor_in = torch.from_numpy(patch_normalized).float().unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pcds_pred = model(tensor_in)
            pred_normalized = pcds_pred[-1].squeeze(0).cpu().numpy().astype(np.float32)

        pred_restored = (pred_normalized * scale) + centroid
        all_repaired_parts.append(pred_restored)
        processed_count += 1

    if not all_repaired_parts:
        print("No patches processed. Check STRIDE or PATCH_SIZE.")
        return

    generated_points = np.vstack(all_repaired_parts).astype(np.float32)
    print(f"Generated points total: \{generated_points.shape[0]\}")

    pcd_gen = o3d.geometry.PointCloud()
    pcd_gen.points = o3d.utility.Vector3dVector(generated_points)

    print("Applying SOR filter...")
    pcd_gen, _ = pcd_gen.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    print(f"Applying Voxel Downsample (\{FINAL_VOXEL_SIZE\})...")
    pcd_gen_ds = pcd_gen.voxel_down_sample(voxel_size=FINAL_VOXEL_SIZE)
    
    print("\uc0\u55356 \u57256  Transferring colors using Fast Weighted KNN (Scipy)...")
    gen_pts = np.asarray(pcd_gen_ds.points)
    
    K_NEIGHBORS = 3
    
    print(f"Building KDTree for \{points_full.shape[0]\} original points...")
    tree = cKDTree(points_full)
    
    print(f"Querying \{K_NEIGHBORS\} nearest neighbors for \{gen_pts.shape[0]\} generated points...")
    dists, indices = tree.query(gen_pts, k=K_NEIGHBORS, workers=-1) 
    
    dists = np.maximum(dists, 1e-8)
    weights = 1.0 / dists
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_norm = weights / weights_sum
    
    neighbor_colors = colors_full[indices]
    weighted_colors = np.sum(neighbor_colors * weights_norm[:, :, np.newaxis], axis=1)
    
    pcd_gen_ds.colors = o3d.utility.Vector3dVector(weighted_colors)
    print("Color transfer complete.")

    print("\uc0\u55357 \u56599  Merging original and generated point clouds...")
    final_pcd = pcd_full + pcd_gen_ds
    o3d.io.write_point_cloud(OUTPUT_PLY, final_pcd)
    print(f"\uc0\u9989  Saved final result to: \{OUTPUT_PLY\}")

if __name__ == "__main__":
    run_inference()}
