import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import math
import time
import matplotlib.pyplot as plt

# model.data_loaderの実装に依存しますが、DataLoaderの引数を上書きできるようにします
from model.data_loader import create_dataloader, load_cfg, find_npz
# 通常のモデル (Self-Conditioningなし) をインポート
from model.model_ffn_gemini import ContinuousDiffusionMLP
from diffusion.diffusion import GaussianDiffusion

# ==========================================
# 1. 高速化設定
# ==========================================
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

class GreedyDiffusionDecoder:
    def __init__(self, model, diffusion, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.timesteps = diffusion.timesteps
        
        # 離散用の事前計算パラメータはここでは使わないため省略

    def decode_batch(self, label_y, num_observables, steps=1, capture_trace=False):
        """
        連続時間DDIMデコーダ (Self-Conditioningなし)
        steps: 推論時のステップ数 (自由に変更可能)
        """
        batch_size = label_y.shape[0]

        # 1. 初期状態 x_T (完全なノイズ)
        x_t = torch.randn((batch_size, num_observables, 1), device=self.device)
        
        # 2. 時間刻みを作成 (1.0 -> 0.0)
        # 例: steps=20 なら [1.0, 0.95, ..., 0.05, 0.0]
        times = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

        trace_xt = []
        if capture_trace:
            trace_xt.append(x_t.detach().cpu().numpy().reshape(batch_size, num_observables))

        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(steps):
                # 現在の時刻 t_now と 次の時刻 t_next
                t_now = times[i]
                t_next = times[i+1]
                
                # バッチ化
                t_batch = torch.full((batch_size,), t_now, device=self.device)
                
                # ==========================================
                # ★ モデル入力用スケール (1000倍)
                # ==========================================
                # 物理的な t (0~1) を、モデルの学習したスケール (0~1000) に変換
                t_input = t_batch * 1000.0

                # モデル予測 (標準入力: x_t, t, y)
                pred_x0 = self.model(x_t, t_input, label_y)
                pred_x0 = torch.clamp(pred_x0, -0.5, 0.5)
                
                # --- DDIM Update (Continuous) ---
                
                # 現在と次の alpha_bar を取得 (物理的な t を使用)
                alpha_bar_now = self.diffusion.get_alpha_bar(t_batch)
                alpha_bar_next = self.diffusion.get_alpha_bar(torch.full((batch_size,), t_next, device=self.device))

                # 形状合わせ (Batch, 1, 1)
                alpha_bar_now = alpha_bar_now.view(batch_size, 1, 1)
                alpha_bar_next = alpha_bar_next.view(batch_size, 1, 1)
                
                # DDIM 更新式 (eta=0, deterministic)
                sigma_now = torch.sqrt(1.0 - alpha_bar_now)
                sigma_next = torch.sqrt(1.0 - alpha_bar_next)
                
                # 現在のノイズ予測 (epsilon) を逆算
                pred_noise = (x_t - torch.sqrt(alpha_bar_now) * pred_x0) / (sigma_now + 1e-8)
                
                # 次のステップの x_t を計算
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + sigma_next * pred_noise

                if capture_trace:
                    trace_xt.append(x_t.detach().cpu().numpy().reshape(batch_size, num_observables))

        # 離散化 (-0.5~0.5 -> 0/1)
        decoded_bits = (x_t > 0.0).float()
        
        if capture_trace:
            return decoded_bits, (trace_xt, [], [])
        else:
            return decoded_bits

def save_debug_plots(trace_data, save_dir="./graph"):
    """
    大量のビットからランダムに間引きしてグラフを描画する
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    trace_xt, _, _ = trace_data
    
    # データを整形
    xt_history = np.array(trace_xt) 
    steps_count, batch_size, num_obs = xt_history.shape
    
    flat_history = xt_history.reshape(steps_count, -1)
    total_available_bits = flat_history.shape[1]
    
    steps = np.arange(steps_count)

    # 描画する本数
    NUM_LINES_TO_PLOT = 2000  

    num_plot = min(NUM_LINES_TO_PLOT, total_available_bits)
    print(f"Plotting {num_plot} random trajectories out of {total_available_bits} total bits...")

    selected_indices = np.random.choice(total_available_bits, num_plot, replace=False)
    subset_data = flat_history[:, selected_indices]

    plt.figure(figsize=(12, 6))
    
    final_values = subset_data[-1]
    colors = np.where(final_values > 0, 'red', 'blue')
    
    for i in range(num_plot):
        plt.plot(steps, subset_data[:, i], color=colors[i], alpha=0.1, linewidth=1)
        
    plt.title(f"Trajectory Sample (Random {num_plot} bits)\nRed: converges to 1, Blue: converges to 0")
    plt.xlabel("Decoding Steps (Reverse Diffusion Process)")
    plt.ylabel("Value of x_t")
    plt.ylim(-2.5, 2.5)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    plt.plot([], [], color='red', label='Bit -> 1')
    plt.plot([], [], color='blue', label='Bit -> 0')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "debug_xt_lines_subsampled.png"))
    plt.close()
    
    print(f"Debug plots saved to {os.path.abspath(save_dir)}")

def parse_args():
    ap = argparse.ArgumentParser(description="QEC Decoding (Optimized)")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--model_path", required=True, help="path to trained model (.pth)")
    ap.add_argument("--batch_size", type=int, default=2048, help="inference batch size")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--npz", nargs="*", default=None, help="npz files for testing")
    ap.add_argument("--compile_mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"], help="torch.compile mode")
    # ステップ数を指定可能に
    ap.add_argument("--steps", type=int, default=20, help="Number of diffusion steps for inference")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg["BATCH_SIZE"] = args.batch_size
    
    if args.npz and len(args.npz) > 0:
        npz_paths = args.npz
    else:
        npz_paths = find_npz(cfg["DATASET_DIR"])
    
    _, _, test_loader = create_dataloader(cfg, npz_paths)
    sample = next(iter(test_loader))
    num_syndrome = sample["label_y"].shape[1]
    num_observables = sample["global_labels"].shape[1]
    
    # モデル構築 (Standard Model)
    timesteps = cfg.get("TIMESTEPS", 200)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device, schedule="cosine")
    
    model = ContinuousDiffusionMLP(
        input_dim=1,
        num_observables=num_observables,
        num_detectors=num_syndrome,
        hidden_size=2048,
        time_emb_dim=256
    ).to(device)

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    # state_dictのキー調整が必要な場合に対応
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.eval()
    
    try:
        model = torch.compile(model, mode=args.compile_mode)
    except:
        pass

    decoder = GreedyDiffusionDecoder(model, diffusion, device)
    
    # Warmup
    dummy_syndrome = torch.zeros((2, num_syndrome), device=device)
    with torch.no_grad():
        decoder.decode_batch(dummy_syndrome, num_observables, steps=args.steps)

    # Decoding loop
    total_samples = 0
    total_bit_errors = 0
    total_logical_errors = 0
    trace_data_storage = None
    
    print(f"\nStarting Decoding (Steps={args.steps})...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Decoding")):
            gt_labels = batch["global_labels"].to(device, non_blocking=True).unsqueeze(-1)
            syndromes = batch["label_y"].to(device, non_blocking=True)
            syndromes = syndromes - 0.5
            
            capture_this_batch = (batch_idx == 0)
            
            if capture_this_batch:
                predicted_bits, trace_data = decoder.decode_batch(
                    syndromes, num_observables, steps=args.steps, capture_trace=True
                )
                trace_data_storage = trace_data
            else:
                predicted_bits = decoder.decode_batch(
                    syndromes, num_observables, steps=args.steps, capture_trace=False
                )

            bit_diff = torch.abs(predicted_bits - gt_labels)
            total_bit_errors += bit_diff.sum().item()
            sample_has_error = bit_diff.view(gt_labels.shape[0], -1).sum(dim=1) > 0
            total_logical_errors += sample_has_error.sum().item()
            total_samples += gt_labels.shape[0]

    elapsed_time = time.time() - start_time
    
    ber = total_bit_errors / (total_samples * num_observables)
    ler = total_logical_errors / total_samples
    accuracy = 1.0 - ler
    throughput = total_samples / elapsed_time
    
    print("\n" + "="*30)
    print(f"Decoding Results (Standard - No Prev)")
    print(f"Time Elapsed        : {elapsed_time:.2f} s")
    print(f"Throughput          : {throughput:.2f} samples/s")
    print("-" * 30)
    print(f"Samples Evaluated   : {total_samples}")
    print(f"Bit Error Rate (BER): {ber:.6f} ({ber*100:.4f}%)")
    print(f"Logical Error Rate  : {ler:.6f} ({ler*100:.4f}%)")
    print(f"Block Accuracy      : {accuracy:.6f} ({accuracy*100:.4f}%)")
    print("="*30 + "\n")

    if trace_data_storage is not None:
        print("Generating visualization plots...")
        save_debug_plots(trace_data_storage)

if __name__ == "__main__":
    main()