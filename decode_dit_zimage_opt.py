import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import time

# ==========================================
# 1. 必要なモジュールのインポート
# ==========================================
from model.data_loader import create_dataloader, load_cfg, find_npz
# 学習に使ったモデル定義 (prev_x0対応版) をインポート
from model.model_dit_zimage_prev import DiffusionTransformer
from diffusion.diffusion import GaussianDiffusion

# ==========================================
# 2. 高速化設定 (TF32 & Benchmark)
# ==========================================
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

class GreedyDiffusionDecoderDDIM:
    """
    Continuous Diffusion Decoder using DDIM & Self-Conditioning
    train_dit_inf.py の学習済みモデルに対応
    """
    def __init__(self, model, diffusion, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        
        # 学習時の最大タイムステップ (Embedderのスケーリングに使用)
        self.timesteps_max = diffusion.timesteps

    def decode_batch(self, label_y, num_observables, steps=1):
        """
        DDIM Sampling
        steps: 推論時のサンプリング分割数 (例: 20, 50, 100)
        """
        batch_size = label_y.shape[0]
     
        # 1. 初期状態 x_T (正規分布)
        x_t = torch.randn((batch_size, num_observables, 1), device=self.device)
        
        # ★ Self-Conditioning用バッファ: 最初はゼロで初期化
        # 学習時と同様、モデル入力時に x_t と concat される
        prev_x0 = torch.zeros_like(x_t)
        
        # 時間スケジュールの作成 (1.0 -> 0.0)
        times = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        #print('steps',steps)
        #print('ditのtimesteps',self.timesteps_max)
        # 2. 逆拡散プロセス (DDIM)
        # AMP (Mixed Precision) コンテキスト内で実行
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(steps):
                # 現在の時刻t と 次の時刻t_next
                t_now = times[i]
                t_next = times[i+1]
                
                # バッチ化
                t_batch = torch.full((batch_size,), t_now, device=self.device)
                t_next_batch = torch.full((batch_size,), t_next, device=self.device)
               # print('t_batch',t_batch)
                # モデルに入力する時間埋め込みは [0, T_max] にスケール変換
                t_input = t_batch * self.timesteps_max
               # print('t_input',t_input)
                # ★ モデル予測 (prev_x0 を入力に使用)
                pred_x0 = self.model(x_t, t_input, label_y, prev_x0=prev_x0)
                
                # クリッピング (データは [-1, 1] に正規化されている前提)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                # ★ 次のステップのために予測値を保存 (Self-Conditioning)
                prev_x0 = pred_x0

                # --- DDIM Update Rule (Deterministic) ---
                # 現在のα_barを取得 (GaussianDiffusion側の計算を利用)
                alpha_bar_now = self.diffusion.get_alpha_bar(t_batch).view(batch_size, 1, 1)
                alpha_bar_next = self.diffusion.get_alpha_bar(t_next_batch).view(batch_size, 1, 1)
                
                # ノイズ予測成分の逆算 (predicted noise)
                # x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * epsilon
                # epsilon = (x_t - sqrt(alpha_bar) * x0) / sqrt(1-alpha_bar)
                sigma_now = torch.sqrt(1.0 - alpha_bar_now)
                pred_noise = (x_t - torch.sqrt(alpha_bar_now) * pred_x0) / (sigma_now + 1e-8)
                
                # 次の x_{t-1} を計算 (sigma_t=0 for DDIM)
                sigma_next = torch.sqrt(1.0 - alpha_bar_next)
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + sigma_next * pred_noise

        # 3. 離散化 (閾値 0.0)
        decoded_bits = (x_t > 0.0).float()
        return decoded_bits

def parse_args():
    ap = argparse.ArgumentParser(description="QEC Decoding DDIM (DiT + Self-Cond)")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--model_path", required=True, help="path to trained model (.pth)")
    ap.add_argument("--batch_size", type=int, default=1024, help="inference batch size")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--npz", nargs="*", default=None, help="npz files for testing")
    ap.add_argument("--mode", type=str, default="in_context")
    # 推論時のステップ数 (学習時のステップ数とは独立して設定可能)
    ap.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    ap.add_argument("--compile_mode", default="reduce-overhead", 
                    choices=["default", "reduce-overhead", "max-autotune"], 
                    help="torch.compile mode")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    
    # バッチサイズ上書き
    cfg["BATCH_SIZE"] = args.batch_size
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from: {cfg['DATASET_DIR']}")
    
    if args.npz and len(args.npz) > 0:
        npz_paths = args.npz
    else:
        npz_paths = find_npz(cfg["DATASET_DIR"])
        
    _, _, test_loader = create_dataloader(cfg, npz_paths)
    
    # 次元数取得
    sample = next(iter(test_loader))
    num_syndrome = sample["label_y"].shape[1]
    num_observables = sample["global_labels"].shape[1]
    
    print(f"Observables: {num_observables}, Syndromes: {num_syndrome}")

    # ==========================================
    # モデル構築
    # ==========================================
    # 学習時のtimesteps (Embedderのスケール用)
    train_timesteps = cfg.get("TIMESTEPS", 200) 
    print(f"Training Timesteps: {train_timesteps}")
    print(f"Inference DDIM Steps: {args.steps}")
    
    if args.mode not in ["in_context", "cross_attn"]:
        print("Warning: Mode not recognized, defaulting to in_context")
        args.mode = "in_context"
    
    print(f"Model Mode: {args.mode}")

    # Diffusionオブジェクト (スケジュール計算用)
    diffusion = GaussianDiffusion(timesteps=train_timesteps, device=device, schedule='cosine')
    
    # モデル定義 (Self-Conditioning対応版)
    model = DiffusionTransformer(
        num_observables=num_observables,
        num_syndrome=num_syndrome,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mode=args.mode,
        dropout=0.1 # 推論時はdropout無効化されるので値は何でも良い
    ).to(device)

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    # State Dictのキー修正 (_orig_mod 等の削除)
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    # コンパイル
    
    print(f"Compiling model with mode='{args.compile_mode}'...")
    try:
        model = torch.compile(model, mode=args.compile_mode)
    except Exception as e:
       print(f"Warning: torch.compile failed ({e}). Running without compilation.") 

    # デコーダの初期化
    decoder = GreedyDiffusionDecoderDDIM(model, diffusion, device)
    
    # --- ウォームアップ ---
    print("Warming up...")
    dummy_syndrome = torch.zeros((2, num_syndrome), device=device)
    with torch.no_grad():
        decoder.decode_batch(dummy_syndrome, num_observables, steps=args.steps)
    print("Warmup finished.")

    # --- 評価ループ ---
    total_samples = 0
    total_bit_errors = 0
    total_logical_errors = 0
    
    print(f"\nStarting Decoding (DDIM Steps={args.steps})...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Decoding"):
            gt_labels = batch["global_labels"].to(device, non_blocking=True).unsqueeze(-1)
            syndromes = batch["label_y"].to(device, non_blocking=True)
            
            # ★ 重要: 学習時と同様に [-1, 1] に正規化
            # train_dit_inf.py では: label_y = 2*label_y.float() - 1.0
            syndromes =  2*syndromes.float() - 1.0
            
            # デコード実行
            predicted_bits = decoder.decode_batch(syndromes, num_observables, steps=args.steps)
            
            # 評価
            bit_diff = torch.abs(predicted_bits - gt_labels)
            total_bit_errors += bit_diff.sum().item()
            
            sample_has_error = bit_diff.view(gt_labels.shape[0], -1).sum(dim=1) > 0
            total_logical_errors += sample_has_error.sum().item()
            total_samples += gt_labels.shape[0]

    elapsed_time = time.time() - start_time
    
    # --- 結果表示 ---
    ber = total_bit_errors / (total_samples * num_observables)
    ler = total_logical_errors / total_samples
    accuracy = 1.0 - ler
    throughput = total_samples / elapsed_time
    
    print("\n" + "="*30)
    print(f"Decoding Results (DDIM Steps={args.steps})")
    print(f"Time Elapsed        : {elapsed_time:.2f} s")
    print(f"Throughput          : {throughput:.2f} samples/s")
    print("-" * 30)
    print(f"Samples Evaluated   : {total_samples}")
    print(f"Bit Error Rate (BER): {ber:.6f} ({ber*100:.4f}%)")
    print(f"Logical Error Rate  : {ler:.6f} ({ler*100:.4f}%)")
    print(f"Block Accuracy      : {accuracy:.6f} ({accuracy*100:.4f}%)")
    print("="*30 + "\n")

if __name__ == "__main__":
    main()