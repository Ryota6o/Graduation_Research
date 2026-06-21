'''×1000をtimesteps に直した '''
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
from model.model_ffn_prevsteps import ContinuousDiffusionMLP
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
        self.betas = diffusion.betas.to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = diffusion.alphas_cumprod.to(device)
        pad = torch.tensor([1.0], device=self.device, dtype=self.alphas_cumprod.dtype)
        self.alphas_cumprod_prev = torch.cat([pad, self.alphas_cumprod[:-1]])
        one_minus_alpha_bar = 1. - self.alphas_cumprod
        self.coef1 = (torch.sqrt(self.alphas_cumprod_prev) * self.betas / one_minus_alpha_bar)
        self.coef2 = (torch.sqrt(self.alphas) * (1. - self.alphas_cumprod_prev) / one_minus_alpha_bar)
        self.t_range = torch.arange(self.timesteps - 1, -1, -1, device=device, dtype=torch.long)
    # steps:推論時の反復回数　ノイズからデータを復元する差異何回にわけてノイズを除去するのかを決める値

    def decode_batch(self, label_y, num_observables, timesteps, steps=200, capture_trace=False):
        batch_size = label_y.shape[0]
        # 1. 初期化
        x_t = torch.randn((batch_size, num_observables, 1), device=self.device)
        # ★ Self-Conditioning用バッファ: 最初はゼロで初期化
        prev_x0 = torch.zeros_like(x_t)
        # tを1から0まで動かす。ステップ数に応じて等間隔に刻む
        times = torch.linspace(1.0, 0.0, steps + 1, device=self.device)
        trace_xt = []
        if capture_trace:
            trace_xt.append(x_t.detach().cpu().numpy().reshape(batch_size, num_observables))
            print(f"DEBUG: decode_batch running with steps={steps}, timesteps={timesteps}")
        # 2. 逆拡散 (DDIM)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(steps):
                # 現在の時刻t
                t_now = times[i]
                # 次に進むべき時刻
                t_next = times[i+1]
                t_batch = torch.full((batch_size,), t_now, device=self.device)
                t_input = t_batch * timesteps
                # ★ モデル入力に prev_x0 を追加 現在の時刻 t_now にいて、そのときのαを計算して、今の時刻と認識させる。
                pred_x0 = self.model(x_t, t_input, label_y, prev_x0=prev_x0)

                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

                # ★ 次のステップのために、今の予測値を prev_x0 として保存 (これがSelf-Conditioningの肝)
                prev_x0 = pred_x0
                # --- DDIM Update (Continuous) ---
                alpha_bar_now = self.diffusion.get_alpha_bar(t_batch).view(batch_size, 1, 1)
                t_next_batch = torch.full((batch_size,), t_next, device=self.device)
                alpha_bar_next = self.diffusion.get_alpha_bar(t_next_batch).view(batch_size, 1, 1)
                sigma_now = torch.sqrt(1.0 - alpha_bar_now)

                pred_noise = (x_t - torch.sqrt(alpha_bar_now) * pred_x0) / (sigma_now + 1e-8)
                sigma_next = torch.sqrt(1.0 - alpha_bar_next)
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + sigma_next * pred_noise
                if capture_trace:
                    trace_xt.append(x_t.detach().cpu().numpy().reshape(batch_size, num_observables))
        decoded_bits = (x_t > 0.0).float()
        if capture_trace:
            return decoded_bits, trace_xt
        else:
            return decoded_bits
def save_debug_plots(trace_data, save_dir="./graph"):
    """
    大量のビットからランダムに間引きしてグラフを描画する
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # データを整形
    # trace_xt の構造: List[Array(Batch, Obs)] -> Array[Steps, Batch, Obs]
    xt_history = np.array(trace_data)
    steps_count, batch_size, num_obs = xt_history.shape
    # 全ビットを1次元に平坦化します (Batch x Obs の区別をなくし、すべてのビットをプールする)
    # shape: [Steps, Total_Bits]
    flat_history = xt_history.reshape(steps_count, -1)
    total_available_bits = flat_history.shape[1]
    steps = np.arange(steps_count)
    # =========================================================
    # ★設定: グラフに描画する線の本数
    # 今後ここを 500 -> 1000 などに変更してください
    # =========================================================
    NUM_LINES_TO_PLOT = 2000
    # =========================================================
    # 指定された数より実際のビット数が少ない場合は、あるだけ全部描く
    num_plot = min(NUM_LINES_TO_PLOT, total_available_bits)
    print(f"Plotting {num_plot} random trajectories out of {total_available_bits} total bits...")
    # ランダムにインデックスを選択 (非復元抽出)
    selected_indices = np.random.choice(total_available_bits, num_plot, replace=False)
    subset_data = flat_history[:, selected_indices]  # shape: [Steps, num_plot]
    # 1. 間引き線グラフ (Subsampled Trajectories)
    plt.figure(figsize=(12, 6))
    # 最終的な値を見て色を決める
    # 最後がプラスなら赤(1)、マイナスなら青(0)
    final_values = subset_data[-1]
    colors = np.where(final_values > 0, 'red', 'blue')
    for i in range(num_plot):
        # alpha=0.1 にすることで、線が重なった部分が濃くなり「分布の濃さ」が見えます
        plt.plot(steps, subset_data[:, i], color=colors[i], alpha=0.1, linewidth=1)
    plt.title(f"Trajectory Sample (Random {num_plot} bits)\nRed: converges to 1, Blue: converges to 0")
    plt.xlabel("Decoding Steps (Reverse Diffusion Process)")
    plt.ylabel("Value of x_t")
    plt.ylim(-2.5, 2.5) # 範囲を少し広めに固定
    plt.grid(True, linestyle='--', alpha=0.4)
    # 凡例用のダミープロット（透明度1.0で作成）
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
    # 拡散プロセスの定義　これを使用して、βやαの累積積などのスケジュールを作成する
    timesteps = cfg.get("TIMESTEPS", 200)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device, schedule='cosine')
    model = ContinuousDiffusionMLP(
        input_dim=1,
        num_observables=num_observables,
        num_detectors=num_syndrome,
        hidden_size=2048,
        time_emb_dim=256
    ).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    try:
        model = torch.compile(model, mode=args.compile_mode)
    except:
        pass
    decoder = GreedyDiffusionDecoder(model, diffusion, device)
    # Warmup
    dummy_syndrome = torch.zeros((2, num_syndrome), device=device)
    with torch.no_grad():
        decoder.decode_batch(dummy_syndrome, num_observables, timesteps)
    # Decoding loop
    total_samples = 0
    total_bit_errors = 0
    total_logical_errors = 0
    trace_data_storage = None
    print("\nStarting Decoding (Optimized)...")
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Decoding")):
            gt_labels = batch["global_labels"].to(device, non_blocking=True).unsqueeze(-1)
            syndromes = batch["label_y"].to(device, non_blocking=True)
            #syndromes = syndromes - 0.5
            syndromes = 2*syndromes - 1.0
            capture_this_batch = (batch_idx == 0)
            if capture_this_batch:
                predicted_bits, trace_data = decoder.decode_batch(
                    syndromes, num_observables, timesteps, capture_trace=True
                )
                trace_data_storage = trace_data
            else:
                predicted_bits = decoder.decode_batch(
                    syndromes, num_observables, timesteps, capture_trace=False
                )
            bit_diff = torch.abs(predicted_bits - gt_labels)
            total_bit_errors += bit_diff.sum().item()
            sample_has_error = bit_diff.view(gt_labels.shape[0], -1).sum(dim=1) > 0
            total_logical_errors += sample_has_error.sum().item()
            total_samples += gt_labels.shape[0]
    elapsed_time = time.time() - start_time
    # --- 結果表示 (詳細表示に変更) ---
    ber = total_bit_errors / (total_samples * num_observables)
    ler = total_logical_errors / total_samples
    accuracy = 1.0 - ler
    throughput = total_samples / elapsed_time
    print("\n" + "="*30)
    print(f"Decoding Results (Optimized)")
    print(f"Time Elapsed        : {elapsed_time:.2f} s")
    print(f"Throughput          : {throughput:.2f} samples/s")
    print("-" * 30)
    print(f"Samples Evaluated   : {total_samples}")
    print(f"Bit Error Rate (BER): {ber:.6f} ({ber*100:.4f}%)")
    print('下を見ろ！' )
    print(f"Logical Error Rate  : {ler:.6f} ({ler*100:.4f}%)")
    print(f"Block Accuracy      : {accuracy:.6f} ({accuracy*100:.4f}%)")
    print("="*30 + "\n")
    if trace_data_storage is not None:
        print("Generating visualization plots...")
        save_debug_plots(trace_data_storage)
if __name__ == "__main__":
    main()