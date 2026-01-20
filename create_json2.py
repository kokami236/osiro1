import json
import os
import glob
from math import floor

# =========================================================
# 
# =========================================================

# train/custom
DATA_ROOT = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/shapenet_pc/02691156/train/custom"

# =========================================================

def main():
    # partial/ai
    partial_root = os.path.join(DATA_ROOT, "partial", "airplane")
    model_dirs = sorted([
        d for d in os.listdir(partial_root)
        if os.path.isdir(os.path.join(partial_root, d))
    ])

    print(f"partial_root: {partial_root}")
    print(f"Found {len(model_dirs)} model dirs.")

    if len(model_dirs) == 0:
        print("moderunasi")
        return

    #
    n = len(model_dirs)
    n_train = floor(n * 0.8)
    n_val   = floor(n * 0.1)

    train_list = model_dirs[:n_train]
    val_list   = model_dirs[n_train:n_train + n_val]
    test_list  = model_dirs[n_train + n_val:]

    print(f"train: {len(train_list)}, val: {len(val_list)}, test: {len(test_list)}")

    # JSON
    json_data = [
        {
            "taxonomy_id": "02691156",
            "taxonomy_name": "airplane",
            "train": train_list,
            "val": val_list,
            "test": test_list
        }
    ]

    #
    output_file = os.path.join(
        "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset",
        "Custom.json"
    )
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"\n seikou!")

if __name__ == "__main__":
    main()
