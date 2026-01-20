import os
import json
import open3d as o3d

ROOT = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/shapenet_pc/02691156/train/custom"
JSON_PATH = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/Custom.json"

with open(JSON_PATH, "r") as f:
    cats = json.load(f)

dc = cats[0]  # airplane
bad = []

def count_points(path):
    if not os.path.exists(path):
        return -1  #
    pc = o3d.io.read_point_cloud(path)
    return len(pc.points)

for split in ["train", "val", "test"]:
    print(f"\n=== split: {split} ===")
    for mid in dc[split]:
        p_path = os.path.join(ROOT, "partial", "airplane", mid, "00.pcd")
        c_path = os.path.join(ROOT, "complete", "airplane", mid + ".pcd")

        n_p = count_points(p_path)
        n_c = count_points(c_path)

        if n_p <= 0 or n_c <= 0:
            print(f"[BAD] {split} {mid}: partial={n_p}, complete={n_c}")
            bad.append((split, mid, n_p, n_c))

print("\nbad sample :", len(bad))
