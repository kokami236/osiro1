import os
import json
import random

# ==========================================
# 
# ==========================================

# PCN + Custom 
BASE_CUSTOM = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/shapenet_pc/02691156/train/custom"

# partial 
PARTIAL_ROOT = os.path.join(BASE_CUSTOM, "partial", "airplane")

#
OUTPUT_JSON = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/Custom.json"

CATEGORY_ID = "02691156"
CATEGORY_NAME = "airplane"

# train / val 
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1   #
RNG_SEED    = 42


def main():
    if not os.path.isdir(PARTIAL_ROOT):
        print(f"? partial: {PARTIAL_ROOT}")
        return

    # ======================================================
    # 1.
    # ======================================================
    model_ids = sorted([
        d for d in os.listdir(PARTIAL_ROOT)
        if os.path.isdir(os.path.join(PARTIAL_ROOT, d))
    ])

    if len(model_ids) == 0:
        print(f"? : {PARTIAL_ROOT}")
        return

    print(f"? : {len(model_ids)}")
    #'02691156-000000', '02691156-000001', ...

    # ======================================================
    # 2. train / val / tes
    # ======================================================
    random.seed(RNG_SEED)
    random.shuffle(model_ids)

    n_total = len(model_ids)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)
    n_test  = n_total - n_train - n_val

    train_ids = model_ids[:n_train]
    val_ids   = model_ids[n_train:n_train + n_val]
    test_ids  = model_ids[n_train + n_val:]

    print(f"  -> train: {len(train_ids)}")
    print(f"  -> val  : {len(val_ids)}")
    print(f"  -> test : {len(test_ids)}")

    # ======================================================
    # 3. Custom.json
    # ======================================================
    json_data = [
        {
            "taxonomy_id": CATEGORY_ID,
            "taxonomy_name": CATEGORY_NAME,
            "train": train_ids,
            "val":   val_ids,
            "test":  test_ids
        }
    ]

    # ======================================================
    # 4.
    # ======================================================
    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"\n? Custom.json -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
