{\rtf1\ansi\ansicpg932\cocoartf2867
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import torch\
import numpy as np\
import open3d as o3d\
import os, sys, math\
from importlib import import_module\
from collections import OrderedDict\
from tqdm import tqdm  # \uc0\u36914 \u25431 \u34920 \u31034 \u29992 \
# \uc0\u9733 \u39640 \u36895 \u21270 \u12398 \u12383 \u12417 \u12395 \u36861 \u21152 \
from scipy.spatial import cKDTree\
\
# \uc0\u23455 \u34892 \u12487 \u12451 \u12524 \u12463 \u12488 \u12522 \u12395  codes \u12501 \u12457 \u12523 \u12480 \u12364 \u12354 \u12427 \u21069 \u25552 \
sys.path.append(os.path.join(os.getcwd(), "codes"))\
\
# \uc0\u35373 \u23450 \u12497 \u12521 \u12513 \u12540 \u12479 \
MODEL_PATH = "/home/limu/seedformer-master/results/train_pcn_Log_2026_01_18_23_14_21/checkpoints/ckpt-best.pth"\
INPUT_PLY  = "/home/limu/seedformer-master/codes/kakegawakesson8.ply"\
OUTPUT_PLY = "final_kakegawa_z_sliced.ply"\
DEVICE = torch.device("cuda:0")\
\
N_INPUT_POINTS = 2048\
\
# \uc0\u12473 \u12521 \u12452 \u12473 \u35373 \u23450 \
PATCH_SIZE = 0.25   # \uc0\u12473 \u12521 \u12452 \u12473 \u12398 \u12300 \u21402 \u12415 \u12301 \
STRIDE     = 0.10   # \uc0\u12473 \u12521 \u12452 \u12473 \u12398 \u12300 \u12378 \u12425 \u12375 \u24133 \u12301 \u65288 \u23567 \u12373 \u12356 \u12392 \u23494 \u12395 \u12394 \u12426 \u12414 \u12377 \u12364 \u26178 \u38291 \u12364 \u12363 \u12363 \u12426 \u12414 \u12377 \u65289 \
\
MIN_KEEP = 50       # \uc0\u23398 \u32722 \u12487 \u12540 \u12479 \u29983 \u25104 \u26178 \u12398  MIN_POINTS \u12395 \u21512 \u12431 \u12379 \u12427 \
JITTER_SIGMA = 0.005\
JITTER_CLIP  = 0.02\
\
# \uc0\u20986 \u21147 \u35373 \u23450 \
FINAL_VOXEL_SIZE = 0.0005\
\
def _upsample_with_jitter(pts, n_points, sigma=0.005, clip=0.02):\
    curr = pts.shape[0]\
    if curr <= 0:\
        return None\
    # \uc0\u26082 \u23384 \u12398 \u28857 \u12363 \u12425 \u12521 \u12531 \u12480 \u12512 \u12395 \u36984 \u25246 \u65288 \u37325 \u35079 \u12354 \u12426 \u65289 \
    idx = np.random.choice(curr, n_points, replace=True)\
    out = pts[idx].copy()\
    if sigma > 0:\
        noise = sigma * np.random.randn(*out.shape)\
        out += np.clip(noise, -clip, clip).astype(np.float32)\
    return out.astype(np.float32)\
\
def run_inference():\
    # \uc0\u12514 \u12487 \u12523 \u12525 \u12540 \u12489 \
    print("Initializing model...")\
    Model = import_module("model")\
    model = Model.dict["seedformer_dim128"](up_factors=[1, 4, 4], num_p0=512).to(DEVICE)\
    \
    if not os.path.exists(MODEL_PATH):\
        print(f"Error: Weights not found at \{MODEL_PATH\}")\
        return\
\
    print("Loading weights...")\
    try:\
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)\
    except TypeError:\
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)\
        \
    original_state_dict = checkpoint["model"]\
    new_state_dict = OrderedDict()\
    for k, v in original_state_dict.items():\
        name = k[7:] if k.startswith("module.") else k\
        new_state_dict[name] = v\
    model.load_state_dict(new_state_dict, strict=True)\
    model.eval()\
\
    # \uc0\u20837 \u21147 \u28857 \u32676 \u35501 \u12415 \u36796 \u12415 \
    print(f"Reading \{INPUT_PLY\}...")\
    pcd_full = o3d.io.read_point_cloud(INPUT_PLY)\
    points_full = np.asarray(pcd_full.points).astype(np.float32)\
    \
    if pcd_full.has_colors():\
        colors_full = np.asarray(pcd_full.colors)\
    else:\
        print("Warning: Input PLY has no colors. Using default gray.")\
        colors_full = np.tile(np.array([0.5, 0.5, 0.5]), (points_full.shape[0], 1))\
        pcd_full.colors = o3d.utility.Vector3dVector(colors_full)\
        \
    print(f"Original points: \{points_full.shape[0]\}")\
\
    # Z\uc0\u36600 \u12473 \u12521 \u12452 \u12473 \u20966 \u29702  (\u23398 \u32722 \u12487 \u12540 \u12479 \u29983 \u25104 \u12525 \u12472 \u12483 \u12463 \u12395 \u28310 \u25312 )\
    mins, maxs = points_full.min(axis=0), points_full.max(axis=0)\
    \
    # Z\uc0\u26041 \u21521 \u12398 \u12473 \u12486 \u12483 \u12503 \u20316 \u25104 \
    z_steps = np.arange(mins[2], maxs[2], STRIDE)\
    total_steps = len(z_steps)\
    \
    print(f"Slice Config: Z-Range[\{mins[2]:.2f\} ~ \{maxs[2]:.2f\}], Steps=\{total_steps\}, PatchSize=\{PATCH_SIZE\}")\
\
    all_repaired_parts = []\
    processed_count = 0\
\
    print("Starting Z-Slice Inference...")\
    \
    # tqdm\uc0\u12391 \u36914 \u25431 \u12496 \u12540 \u12434 \u34920 \u31034 \
    for z in tqdm(z_steps, desc="Processing Slices"):\
        \
        # \uc0\u12473 \u12521 \u12452 \u12473 \u31684 \u22258 \u12398 \u23450 \u32681 \
        z_bottom = z\
        z_top = z + PATCH_SIZE\
        \
        # Numpy\uc0\u12398 \u12502 \u12540 \u12523 \u12452 \u12531 \u12487 \u12483 \u12463 \u12473 \u12391 \u39640 \u36895 \u12395 \u25277 \u20986 \
        # Z\uc0\u24231 \u27161 \u12364 \u31684 \u22258 \u20869 \u12395 \u12354 \u12427 \u28857 \u12384 \u12369 \u12434 \u21462 \u24471  (XY\u12399 \u21046 \u38480 \u12394 \u12375 \u65309 \u36650 \u20999 \u12426 )\
        mask = (points_full[:, 2] >= z_bottom) & (points_full[:, 2] < z_top)\
        patch_points = points_full[mask]\
\
        # \uc0\u28857 \u25968 \u12364 \u23569 \u12394 \u12377 \u12366 \u12427 \u22580 \u21512 \u12399 \u12473 \u12461 \u12483 \u12503 \
        if patch_points.shape[0] < MIN_KEEP:\
            continue\
\
        # Sampling (2048\uc0\u28857 \u12395 \u25539 \u12360 \u12427 )\
        if patch_points.shape[0] >= N_INPUT_POINTS:\
            sel = np.random.choice(patch_points.shape[0], N_INPUT_POINTS, replace=False)\
            patch_points = patch_points[sel].astype(np.float32)\
        else:\
            patch_points = _upsample_with_jitter(\
                patch_points.astype(np.float32), \
                N_INPUT_POINTS, \
                sigma=JITTER_SIGMA, \
                clip=JITTER_CLIP\
            )\
\
        # \uc0\u27491 \u35215 \u21270  & \u25512 \u35542 \
        # \uc0\u37325 \u24515 \u35336 \u31639 \u65288 \u23398 \u32722 \u12487 \u12540 \u12479 \u12418  np.mean \u12391 \u20013 \u24515 \u21270 \u12375 \u12390 \u12356 \u12427 \u12383 \u12417 \u12371 \u12428 \u12395 \u21512 \u12431 \u12379 \u12427 \u65289 \
        centroid = np.mean(patch_points, axis=0, dtype=np.float32)\
        patch_centered = patch_points - centroid\
        \
        # \uc0\u12473 \u12465 \u12540 \u12523 \u27491 \u35215 \u21270 \
        scale = np.max(np.linalg.norm(patch_centered, axis=1)) + 1e-8\
        patch_normalized = patch_centered / scale\
\
        # Tensor\uc0\u21270 \
        tensor_in = torch.from_numpy(patch_normalized).float().unsqueeze(0).to(DEVICE)\
\
        # \uc0\u25512 \u35542 \
        with torch.no_grad():\
            pcds_pred = model(tensor_in)\
            # SeedFormer\uc0\u12399 \u12522 \u12473 \u12488 \u12391 \u20986 \u21147 \u12373 \u12428 \u12427 \u22580 \u21512 \u12364 \u12354 \u12427 \u12383 \u12417 \u26368 \u24460 \u12398 \u35201 \u32032 (Fine)\u12434 \u21462 \u24471 \
            pred_normalized = pcds_pred[-1].squeeze(0).cpu().numpy().astype(np.float32)\
\
        # Restore (\uc0\u24231 \u27161 \u24489 \u20803 )\
        pred_restored = (pred_normalized * scale) + centroid\
        all_repaired_parts.append(pred_restored)\
        processed_count += 1\
\
    if not all_repaired_parts:\
        print("No patches processed. Check STRIDE or PATCH_SIZE.")\
        return\
\
    # \uc0\u29983 \u25104 \u32080 \u26524 \u12398 \u32113 \u21512 \
    generated_points = np.vstack(all_repaired_parts).astype(np.float32)\
    print(f"Generated points total: \{generated_points.shape[0]\}")\
\
    pcd_gen = o3d.geometry.PointCloud()\
    pcd_gen.points = o3d.utility.Vector3dVector(generated_points)\
\
    print("Applying SOR filter...")\
    pcd_gen, _ = pcd_gen.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)\
\
    print(f"Applying Voxel Downsample (\{FINAL_VOXEL_SIZE\})...")\
    pcd_gen_ds = pcd_gen.voxel_down_sample(voxel_size=FINAL_VOXEL_SIZE)\
    \
    # \uc0\u33394 \u24773 \u22577 \u12398 \u36578 \u20889  (Scipy cKDTree\u29256 )\
    print("\uc0\u55356 \u57256  Transferring colors using Fast Weighted KNN (Scipy)...")\
    gen_pts = np.asarray(pcd_gen_ds.points)\
    \
    # K\uc0\u36817 \u20621 \
    K_NEIGHBORS = 3\
    \
    print(f"Building KDTree for \{points_full.shape[0]\} original points...")\
    tree = cKDTree(points_full)\
    \
    print(f"Querying \{K_NEIGHBORS\} nearest neighbors for \{gen_pts.shape[0]\} generated points...")\
    dists, indices = tree.query(gen_pts, k=K_NEIGHBORS, workers=-1) \
    \
    # \uc0\u37325 \u12415 \u35336 \u31639  (IDW)\
    dists = np.maximum(dists, 1e-8)\
    weights = 1.0 / dists\
    weights_sum = np.sum(weights, axis=1, keepdims=True)\
    weights_norm = weights / weights_sum\
    \
    neighbor_colors = colors_full[indices]\
    weighted_colors = np.sum(neighbor_colors * weights_norm[:, :, np.newaxis], axis=1)\
    \
    pcd_gen_ds.colors = o3d.utility.Vector3dVector(weighted_colors)\
    print("Color transfer complete.")\
\
    # \uc0\u12510 \u12540 \u12472 \u12375 \u12390 \u20445 \u23384 \
    print("\uc0\u55357 \u56599  Merging original and generated point clouds...")\
    final_pcd = pcd_full + pcd_gen_ds\
    o3d.io.write_point_cloud(OUTPUT_PLY, final_pcd)\
    print(f"\uc0\u9989  Saved final result to: \{OUTPUT_PLY\}")\
\
if __name__ == "__main__":\
    run_inference()}