import json
import os
import glob

# =========================================================
# ▼ 設定：自分の環境に合わせて書き換えてください
# =========================================================

# データのルートフォルダ (datasetフォルダのパス)
DATA_ROOT = "/home/limu/seedformer-master/My_PCN_Dataset-20251204T033849Z-1-001/My_PCN_Dataset/shapenet_pc/02691156"

# カテゴリ名（フォルダ名と同じにする）
CATEGORY_NAME = "custom" 

# =========================================================

def get_filenames(subset):
    # partialフォルダの中にある .pcd ファイル名を取得する
    # 検索パス: dataset/train/custom/partial/*.pcd
    search_path = os.path.join(DATA_ROOT, subset, CATEGORY_NAME, 'partial', '02691156-*.pcd')
    files = glob.glob(search_path)
    
    # ファイルが見つかったか確認
    print(f"Searching in: {search_path}")
    print(f"Found {len(files)} files.")
    
    # パスからファイル名（拡張子なし）だけを取り出してリストにする
    # 例: /path/to/data001.pcd -> data001
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])

def main():
    # 学習データとテストデータのファイル名を取得
    train_list = get_filenames('train')
    test_list = get_filenames('test')

    if len(train_list) == 0:
        print("【エラー】学習データが見つかりませんでした。パスやフォルダ名を確認してください。")
        return

    # JSONの構造を作成
    json_data = [
        {
            "taxonomy_id": "02691156",  # これがフォルダ名と一致する必要がある
            "taxonomy_name": "airplane",
            "train": train_list,
            "test": test_list,
            "val": test_list  # val（検証用）はtestと同じでOK
        }
    ]

    # ファイルに書き出し
    output_file = "Custom.json"
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"\n成功！ '{output_file}' を作成しました。")

if __name__ == "__main__":
    main()
