import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import time
import csv
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. 必要なモジュールのインポート
# ==========================================
from model.data_loader import create_dataloader, load_cfg, find_npz
# 学習に使ったモデル定義 (prev_x0対応版) をインポート
try:
    from model.model_dit_zimage_prev import DiffusionTransformer
except ImportError:
    from model.model_dit_zimage import DiffusionTransformer

from diffusion.diffusion import GaussianDiffusion

# ==========================================
# 2. 高速化設定
# ==========================================
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

class GreedyDiffusionDecoderDDIM:
    """
    Continuous Diffusion Decoder using DDIM & Self-Conditioning
    decode_dit_zimage_opt.py と同じロジック
    """
    def __init__(self, model, diffusion, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.timesteps_max = diffusion.timesteps

    def decode_batch(self, label_y, num_observables, steps):
        batch_size = label_y.shape[0]

        # 1. 初期状態
        x_t = torch.randn((batch_size, num_observables, 1), device=self.device)
        prev_x0 = torch.zeros_like(x_t)
        
        # 時間スケジュールの作成
        times = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

        # 2. 逆拡散プロセス (DDIM)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(steps):
                t_now = times[i]
                t_next = times[i+1]
                
                t_batch = torch.full((batch_size,), t_now, device=self.device)
                t_next_batch = torch.full((batch_size,), t_next, device=self.device)
                
                t_input = t_batch * self.timesteps_max

                # モデル予測
                pred_x0 = self.model(x_t, t_input, label_y, prev_x0=prev_x0)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                prev_x0 = pred_x0

                # DDIM Update
                alpha_bar_now = self.diffusion.get_alpha_bar(t_batch).view(batch_size, 1, 1)
                alpha_bar_next = self.diffusion.get_alpha_bar(t_next_batch).view(batch_size, 1, 1)
                
                sigma_now = torch.sqrt(1.0 - alpha_bar_now)
                pred_noise = (x_t - torch.sqrt(alpha_bar_now) * pred_x0) / (sigma_now + 1e-8)
                
                sigma_next = torch.sqrt(1.0 - alpha_bar_next)
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + sigma_next * pred_noise

        # 3. 離散化
        decoded_bits = (x_t > 0.0).float()
        return decoded_bits

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    ap = argparse.ArgumentParser(description="QEC Decoding Sweep (DiT DDIM)")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--model_path", required=True, help="path to trained model (.pth)")
    ap.add_argument("--batch_size", type=int, default=1024, help="inference batch size")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--npz", nargs="*", default=None, help="npz files for testing")
    ap.add_argument("--mode", type=str, default="in_context")
    ap.add_argument("--compile_mode", default="reduce-overhead", 
                    choices=["default", "reduce-overhead", "max-autotune"], 
                    help="torch.compile mode")
    ap.add_argument("--output_csv", default="ler_vs_steps_dit.csv", help="Output filename for CSV table")
    ap.add_argument("--output_graph", default="dit_144.png", help="Output filename for graph")
    ap.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    return ap.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    cfg = load_cfg(args.config)
    cfg["BATCH_SIZE"] = args.batch_size
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data from: {cfg['DATASET_DIR']}")
    
    if args.npz and len(args.npz) > 0:
        npz_paths = args.npz
    else:
        npz_paths = find_npz(cfg["DATASET_DIR"])
        
    _, _, test_loader = create_dataloader(cfg, npz_paths)
    
    sample = next(iter(test_loader))
    num_syndrome = sample["label_y"].shape[1]
    num_observables = sample["global_labels"].shape[1]
    
    if args.mode not in ["in_context", "cross_attn"]:
        print("Warning: Mode not recognized, defaulting to in_context")
        args.mode = "in_context"

    model = DiffusionTransformer(
        num_observables=num_observables,
        num_syndrome=num_syndrome,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mode=args.mode,
        dropout=0.1
    ).to(device)

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    try:
        model = torch.compile(model, mode=args.compile_mode)
    except:
        pass

    train_timesteps = cfg.get("TIMESTEPS", 200)
    diffusion = GaussianDiffusion(timesteps=train_timesteps, device=device, schedule='cosine')
    decoder = GreedyDiffusionDecoderDDIM(model, diffusion, device)

    # 200 から 10 まで 10刻み、最後に 1 を追加
    step_list = list(range(200, 9, -10))
    step_list.append(1)
    
    print(f"\nTarget Sampling Steps List: {step_list}")

    results_steps = []
    results_ler = []
    results_ber = []
    results_throughput = []

    # ウォームアップ
    print("Warming up...")
    dummy_syndrome = torch.zeros((2, num_syndrome), device=device)
    with torch.no_grad():
        decoder.decode_batch(dummy_syndrome, num_observables, steps=10)

    # スイープループ
    for steps in step_list:
        print(f"\n--- Steps = {steps} ---")
        total_samples = 0
        total_bit_errors = 0
        total_logical_errors = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Dec (Steps={steps})"):
                gt_labels = batch["global_labels"].to(device, non_blocking=True).unsqueeze(-1)
                syndromes = batch["label_y"].to(device, non_blocking=True)
                
                # 正規化
                syndromes = 2 * syndromes.float() - 1.0
                
                predicted_bits = decoder.decode_batch(syndromes, num_observables, steps=steps)
                
                bit_diff = torch.abs(predicted_bits - gt_labels)
                total_bit_errors += bit_diff.sum().item()
                sample_has_error = bit_diff.view(gt_labels.shape[0], -1).sum(dim=1) > 0
                total_logical_errors += sample_has_error.sum().item()
                total_samples += gt_labels.shape[0]

        elapsed_time = time.time() - start_time
        ber = total_bit_errors / (total_samples * num_observables)
        ler = total_logical_errors / total_samples
        throughput = total_samples / elapsed_time
        
        results_steps.append(steps)
        results_ler.append(ler)
        results_ber.append(ber)
        results_throughput.append(throughput)
        print(f" -> LER = {ler:.6f}, BER = {ber:.6f}, Speed = {throughput:.1f} samp/s")

    # CSV保存
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "LER", "BER", "Throughput"])
        for i in range(len(results_steps)):
            writer.writerow([results_steps[i], results_ler[i], results_ber[i], results_throughput[i]])
    print(f"\nResults saved to {args.output_csv}")

    # ==========================================
    # グラフ描画 (フォントサイズ変更済み)
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # プロット
    plt.plot(results_steps, results_ler, marker='o', linestyle='-', color='b', label='DiT DDIM Performance')
    
    # ★変更点: フォントサイズを大きく設定 (fontsize=18, labelsize=16)
    plt.xlabel('Sampling Steps (DDIM)', fontsize=18)
    plt.ylabel('Logical Error Rate (LER)', fontsize=18)
    plt.title('Logical Error Rate vs. Sampling Steps (DiT)', fontsize=20)
    
    # 目盛りの数字を大きくする
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.grid(True, which="both", ls="--")
    plt.xscale('log')
    
    ticks = [1, 10, 20, 50, 100, 200]
    plt.xticks(ticks, [str(t) for t in ticks])
    
    # 凡例も少し大きく
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(args.output_graph)
    print(f"Graph saved to {args.output_graph}")

if __name__ == "__main__":
    main()