import os
import glob
import yaml
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# --- config.yaml の読み込み ---
def load_cfg(path="config.yaml"):
    """ config.yaml を読み込み、辞書型で返す """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# --- .npz データセット ---
class QECNpzDataset(Dataset):
    """QEC用 .npz ファイルをロードする Datasetクラス"""
    def __init__(self, npz_paths):
        # 複数の .npz ファイルを読み込んで結合
        #arrs = [np.load(p) for p in npz_paths]
        arrs = []
        valid_paths = []
        for p in npz_paths:
            try:
        # 試しに読み込んでみる
                data = np.load(p, allow_pickle=True)
        # 中身にアクセスできるか確認 (破損しているとここで落ちる)
                _ = data.files 

                arrs.append(data)
                valid_paths.append(p)
            except Exception as e:
                print(f"⚠️ Warning: Skipping corrupted file: {p}")
        # print(f"   Reason: {e}")
        print(f"Loading {len(npz_paths)} files...")

        print(f"arrs[0]['global_labels'] shape: {arrs[0]['global_labels'].shape}")  # デバッグ: 最初のファイルの形状確認
        print(f"arrs[0]['label_y'] shape: {arrs[0]['label_y'].shape}")  # デバッグ: label_y の形状確認
        
        self.g   = torch.from_numpy(np.concatenate([a["global_labels"]  for a in arrs], axis=0)).float()
        self.label_y  = torch.from_numpy(arrs[0]["label_y"]).float()
    
    def __len__(self):
        return self.g.shape[0]

    def __getitem__(self, i):
        return {
            "global_labels": self.g[i],
            "label_y": self.label_y[i],
        }

# --- .npz データ探索 ---
def find_npz(dataset_dir):
    """指定ディレクトリから .npz ファイルを探索してリストで返す"""
    paths = sorted(glob.glob(os.path.join(dataset_dir, "bbcode_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no npz in {dataset_dir}")
    return paths

# --- DataLoader 作成 ---
def create_dataloader(cfg, npz_paths, num_workers=2):
    """DataLoader を作成して返す"""
    # .npzデータをまとめて読み込み、Datasetに変換
    ds = QECNpzDataset(npz_paths)

    # データセットを訓練・検証・テストに分割
    train_ds, valid_ds, test_ds = split_dataset(
        ds, cfg["TRAIN_SPLIT"], cfg["VALID_SPLIT"], cfg["TEST_SPLIT"], cfg["SEED"]
    )

    # GPU使用時に pin_memory=True で高速化
    pin = (str(cfg.get("DEVICE", "cpu")).lower() == "cuda")
    train_loader = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"], shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["BATCH_SIZE"], shuffle=False,
                             num_workers=num_workers, pin_memory=pin, drop_last=True)

    return train_loader, valid_loader, test_loader

# --- データセットの分割 ---
def split_dataset(ds: Dataset, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int):
    """データセットを訓練・検証・テスト用に分割"""
    total = len(ds)
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("TRAIN_SPLIT + VALID_SPLIT + TEST_SPLIT must be 1.0")

    train_len = int(total * train_ratio)
    valid_len = int(total * valid_ratio)
    test_len = total - train_len - valid_len

    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [train_len, valid_len, test_len], generator=gen)
