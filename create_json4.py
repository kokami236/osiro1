import os
import json
import random

# PCN + Custom
BASE_CUSTOM = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/shapenet_pc/02691156/train/custom"

PARTIAL_ROOT  = os.path.join(BASE_CUSTOM, "partial",  "airplane")
COMPLETE_ROOT = os.path.join(BASE_CUSTOM, "complete", "airplane")

OUTPUT_JSON = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/Custom.json"

CATEGORY_ID = "02691156"
CATEGORY_NAME = "airplane"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
RNG_SEED    = 42

# partial に最低限揃っていてほしい view
REQUIRED_PARTIAL = [f"{i:02d}.pcd" for i in range(5)]  # 00〜04

def main():
    if not os.path.isdir(PARTIAL_ROOT):
        print(f"[ERR] partial not found: {PARTIAL_ROOT}")
        return
    if not os.path.isdir(COMPLETE_ROOT):
        print(f"[ERR] complete not found: {COMPLETE_ROOT}")
        return

    # 1) partialのmodel_id一覧
    partial_ids = sorted([
        d for d in os.listdir(PARTIAL_ROOT)
        if os.path.isdir(os.path.join(PARTIAL_ROOT, d))
    ])
    if len(partial_ids) == 0:
        print(f"[ERR] no partial dirs: {PARTIAL_ROOT}")
        return

    # 2) completeのmodel_id集合（<id>.pcd）
    complete_ids = set()
    for fn in os.listdir(COMPLETE_ROOT):
        if fn.endswith(".pcd"):
            complete_ids.add(fn[:-4])  # remove ".pcd"

    # 3) 両方揃ってる + partialが00-04全部あるものだけ通す
    usable = []
    missing_gt = 0
    missing_views = 0

    for mid in partial_ids:
        gt_path = os.path.join(COMPLETE_ROOT, f"{mid}.pcd")
        if mid not in complete_ids or (not os.path.exists(gt_path)):
            missing_gt += 1
            continue

        pdir = os.path.join(PARTIAL_ROOT, mid)
        ok = True
        for r in REQUIRED_PARTIAL:
            if not os.path.exists(os.path.join(pdir, r)):
                ok = False
                break
        if not ok:
            missing_views += 1
            continue

        usable.append(mid)

    print(f"[INFO] partial dirs            : {len(partial_ids)}")
    print(f"[INFO] complete files          : {len(complete_ids)}")
    print(f"[INFO] usable (both + 00-04)    : {len(usable)}")
    print(f"[INFO] filtered (missing gt)    : {missing_gt}")
    print(f"[INFO] filtered (missing views) : {missing_views}")

    if len(usable) == 0:
        print("[ERR] usable が 0。フォルダ構造かパス設定を再確認して。")
        return

    # 4) split
    random.seed(RNG_SEED)
    random.shuffle(usable)

    n_total = len(usable)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_ids = usable[:n_train]
    val_ids   = usable[n_train:n_train + n_val]
    test_ids  = usable[n_train + n_val:]

    print(f"  -> train: {len(train_ids)}")
    print(f"  -> val  : {len(val_ids)}")
    print(f"  -> test : {len(test_ids)}")

    # 5) Custom.json 書き出し
    json_data = [{
        "taxonomy_id": CATEGORY_ID,
        "taxonomy_name": CATEGORY_NAME,
        "train": train_ids,
        "val":   val_ids,
        "test":  test_ids
    }]

    with open(OUTPUT_JSON, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"\n[OK] Custom.json -> {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
