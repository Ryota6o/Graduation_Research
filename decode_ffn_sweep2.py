# 降順でデコードをする。 #

import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import csv

# ==========================================
# 重要: heatmap側と同じモデル定義を使用する必要があります
# モデルが prev_x0 引数を受け取る必要があるためです
# ==========================================
# もし model_ffn_gemini が prev_x0 に対応していない場合は 
# model_ffn_prevsteps に変更してください。ここでは heatmap に合わせます。
try:
    from model.model_ffn_prevsteps import ContinuousDiffusionMLP
except ImportError:
    # ファイルがない場合のフォールバック（環境に合わせて調整してください）
    from model.model_ffn_gemini import ContinuousDiffusionMLP

from model.data_loader import create_dataloader, load_cfg, find_npz
from diffusion.diffusion import GaussianDiffusion

# ==========================================
# 1. 高速化設定
# ==========================================
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.set_float32_matmul_precision('high')

class GreedyDiffusionDecoderDDIM:
    """
    heatmap_prev.py と同じ DDIM + Self-Conditioning 実装
    """
    def __init__(self, model, diffusion, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        # 学習時の最大タイムステップ (例: 1000)
        self.timesteps_max = diffusion.timesteps 

    def decode_batch(self, label_y, num_observables, steps):
        """
        steps: 推論時のサンプリング回数 (例: 10, 50, 200)
        """
        batch_size = label_y.shape[0]

        # 1. 初期化 x_T (正規分布)
        x_t = torch.randn((batch_size, num_observables, 1), device=self.device)
        
        # Self-Conditioning用バッファ: 最初はゼロで初期化
        prev_x0 = torch.zeros_like(x_t)
        
        # 時間の刻みを作成 (1.0 -> 0.0)
        times = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

        # 2. 逆拡散 (DDIM)
        # AMP (Mixed Precision) コンテキスト内で実行
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for i in range(steps):
                # 現在の時刻tと次の時刻t_next
                t_now = times[i]
                t_next = times[i+1]
                
                # バッチ化
                t_batch = torch.full((batch_size,), t_now, device=self.device)
                
                # モデルに入力する時間は [0, T_max] にスケール変換
                t_input = t_batch * self.timesteps_max

                # モデル予測 (prev_x0 を入力に使用)
                # ※ heatmap_prev.py のロジックに準拠
                pred_x0 = self.model(x_t, t_input, label_y, prev_x0=prev_x0)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
                
                # 次のステップのために予測値を保存 (Self-Conditioning)
                prev_x0 = pred_x0

                # --- DDIM Update Rule ---
                # 現在のα_barを取得
                alpha_bar_now = self.diffusion.get_alpha_bar(t_batch).view(batch_size, 1, 1)
                # 次の時刻のα_barを取得
                t_next_batch = torch.full((batch_size,), t_next, device=self.device)
                alpha_bar_next = self.diffusion.get_alpha_bar(t_next_batch).view(batch_size, 1, 1)
                
                # ノイズ予測成分の逆算
                sigma_now = torch.sqrt(1.0 - alpha_bar_now)
                pred_noise = (x_t - torch.sqrt(alpha_bar_now) * pred_x0) / (sigma_now + 1e-8)
                
                # 次の x_{t-1} を計算
                sigma_next = torch.sqrt(1.0 - alpha_bar_next)
                x_t = torch.sqrt(alpha_bar_next) * pred_x0 + sigma_next * pred_noise

        # 3. 離散化
        decoded_bits = (x_t > 0.0).float()
        return decoded_bits

def parse_args():
    ap = argparse.ArgumentParser(description="QEC Decoding Sweep (DDIM Steps)")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--model_path", required=True, help="path to trained model (.pth)")
    ap.add_argument("--batch_size", type=int, default=2048, help="inference batch size")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--npz", nargs="*", default=None, help="npz files for testing")
    ap.add_argument("--compile_mode", default="reduce-overhead", 
                    choices=["default", "reduce-overhead", "max-autotune"], 
                    help="torch.compile mode")
    ap.add_argument("--output_graph", default="ffn_72_600.png", help="Output filename for graph")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # バッチサイズ上書き
    cfg["BATCH_SIZE"] = args.batch_size
    
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

    # モデル構築
    model = ContinuousDiffusionMLP(
        input_dim=1,
        num_observables=num_observables,
        num_detectors=num_syndrome,
        hidden_size=2048,
        time_emb_dim=256
    ).to(device)

    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # コンパイル
    try:
        model = torch.compile(model, mode=args.compile_mode)
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}). Running without compilation.")

    # ==========================================
    # 拡散プロセスの初期化 (学習時の設定を使用)
    # ==========================================
    # DDIMでは diffusion オブジェクト自体は固定し、
    # decode_batch に渡す steps を変化させます。
    train_timesteps = cfg.get("TIMESTEPS", 200) # デフォルト1000など
    print(f"Training Timesteps (Fixed): {train_timesteps}")
    diffusion = GaussianDiffusion(timesteps=train_timesteps, device=device)

    print('diffusionのtime steps',diffusion.timesteps)
    
    decoder = GreedyDiffusionDecoderDDIM(model, diffusion, device)

    # ==========================================
    # ステップ数のスイープ設定 (変更箇所)
    # ==========================================
    # 要望: 200から10まで10刻み、最後にT=1
    # range(start, stop, step) -> 200, 190, ..., 10 (stop=9にする)
    step_list = list(range(200, 9, -10))
    step_list.append(1) # 最後に1を追加
    
    print(f"\nTarget Sampling Steps List: {step_list}")
    
    results_steps = []
    results_ler = []
    
    # ==========================================
    # スイープループ
    # ==========================================
    for steps in step_list:
        print(f"\n" + "="*40)
        print(f" Running Decoding for Steps = {steps}")
        print("="*40)
        
        total_samples = 0
        total_bit_errors = 0
        total_logical_errors = 0
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Dec (Steps={steps})"):
                gt_labels = batch["global_labels"].to(device, non_blocking=True).unsqueeze(-1)
                syndromes = batch["label_y"].to(device, non_blocking=True)
                
                # heatmapに合わせて正規化を修正 (-0.5 か 2x-1 か、学習時の設定に合わせる)
                # decode_ffn_heatmap_prev.py では 2*syndromes - 1.0 でした
                syndromes = 2 * syndromes - 1.0
                
                # デコード実行 (stepsを渡す)
                predicted_bits = decoder.decode_batch(syndromes, num_observables, steps=steps)
                
                # 評価
                bit_diff = torch.abs(predicted_bits - gt_labels)
                total_bit_errors += bit_diff.sum().item()
                
                # LER評価
                sample_has_error = bit_diff.view(gt_labels.shape[0], -1).sum(dim=1) > 0
                total_logical_errors += sample_has_error.sum().item()
                total_samples += gt_labels.shape[0]

        elapsed = time.time() - start_time
        
        # 結果計算
        ler = total_logical_errors / total_samples
        ber = total_bit_errors / (total_samples * num_observables)
        
        results_steps.append(steps)
        results_ler.append(ler)
        
        print(f" -> Steps={steps}: LER = {ler:.6f}, BER = {ber:.6f} ({elapsed:.2f}s)")

    # ==========================================
    # 結果の保存と描画
    # ==========================================
    csv_filename = "ler_vs_steps_ddim.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Steps", "LER"])
        for s, l in zip(results_steps, results_ler):
            writer.writerow([s, l])
    print(f"\nResults saved to {csv_filename}")

    # グラフ描画
    plt.figure(figsize=(10, 6))
    plt.plot(results_steps, results_ler, marker='o', linestyle='-', color='b', label='DDIM Performance')
    
    plt.xlabel('Sampling Steps (Reverse Process)',fontsize=18)
    plt.ylabel('Logical Error Rate (LER)',fontsize=18)
    plt.title('Logical Error Rate vs. DDIM Sampling Steps',fontsize=18)
    plt.title('Logical Error Rate vs. Sampling Steps (DiT)', fontsize=20)
    plt.grid(True, which="both", ls="--")
    
    # 軸を反転させるか、ログスケールにするかはお好みで
    # ステップ数が少ない方が右側に来るようにしたい場合は plt.gca().invert_xaxis()
    plt.xscale('log')
    
    ticks = [1, 10, 20, 50, 100, 200]
    plt.xticks(ticks, [str(t) for t in ticks])
    
    plt.legend(fontsize=14)
    plt.savefig(args.output_graph)
    print(f"Graph saved to {args.output_graph}")
    print("Done.")

if __name__ == "__main__":
    main()