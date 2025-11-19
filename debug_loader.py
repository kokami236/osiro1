import sys
import os
import torch
import traceback

# 既存のコードが codes フォルダにあるため、パスを追加
sys.path.append(os.path.join(os.getcwd(), 'codes'))

from train_shapenet55 import cfg 
from data_utils import load_dataset_and_loader

print("Configuration loaded. Attempting to load DataLoader...")

try:
    # データローダーをロード
    train_data_loader, val_data_loader = load_dataset_and_loader(cfg) 
    print("DataLoader loaded successfully. Attempting to get first batch...")

    # 最初のバッチを手動で取得 (ここでI/Oの問題が表面化する)
    data = next(iter(train_data_loader))
    
    print("SUCCESS: First batch retrieved successfully!")
    print("Data keys:", data.keys())

except Exception as e:
    print(f"ERROR: An exception occurred during loading: {e}")
    traceback.print_exc()

print("Script finished.")