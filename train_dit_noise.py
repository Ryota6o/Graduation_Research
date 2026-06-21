'''無限学習 + 定期保存 + ベスト保存 + レジューム機能 + ReduceLROnPlateau
   (Noise Prediction Version)  Prev機能がついいるる
'''

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
from time import time
import gc
import yaml
import re
import math

# --- 必要なライブラリのインポート (Stim, QLDPC) ---
import stim
import sympy
from qldpc import codes
from qldpc.objects import Pauli

# --- モデル定義のインポート ---
# model_dit_zimage_prev.py が model/model_dit_zimage.py にあると仮定
# クラス名が ContinuousDiffusionMLP ではなく DiffusionTransformer のようなので修正してインポート
try:
    from model.model_dit_zimage_prev import DiffusionTransformer as ContinuousDiffusionMLP
except ImportError:
    # ファイル名が違う場合のフォールバック（ユーザー環境に合わせて調整してください）
    from model.model_ffn_prevsteps import ContinuousDiffusionMLP 

from diffusion.diffusion import GaussianDiffusion

torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high')

# ==========================================
# 0. ノイズ予測用のヘルパー関数 (新規追加)
# ==========================================
def predict_start_from_noise(x_t, t, noise, diffusion):
    """
    ノイズ予測値から x_0 を逆算する関数
    x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
    """
    alpha_bar = diffusion.get_alpha_bar(t).to(x_t.device)
    shape = [x_t.shape[0]] + [1] * (x_t.ndim - 1)
    alpha_bar = alpha_bar.view(*shape)

    sqrt_alpha_bar = torch.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar)
    
    # ゼロ除算回避（t=1.0付近での安定性のため）
    sqrt_alpha_bar = torch.clamp(sqrt_alpha_bar, min=1e-5)
    
    x_0_pred = (x_t - sqrt_one_minus_alpha_bar * noise) / sqrt_alpha_bar
    return x_0_pred

# ==========================================
# 1. データ生成ロジック (変更なし)
# ==========================================
def make_bbcode(n0, k0, n1=None, k1=None, poly_a=None, poly_b=None):
    if n1 is None: n1 = n0
    if k1 is None: k1 = k0
    x, y = sympy.symbols('x y')
    if poly_a is None:
        poly_a = x**3 + y + y**2
    if poly_b is None:
        poly_b = y**3 + x + x**2
    code = codes.BBCode([n0, k0], poly_a=poly_a, poly_b=poly_b)
    return code

def get_mpp_targets(matrix, pauli_type):
    if not isinstance(matrix, np.ndarray): matrix = matrix.toarray()
    instructions = []
    for row in matrix:
        qubits = np.where(row)[0]
        if len(qubits) == 0: continue
        targets = []
        for q in qubits:
            if pauli_type == 'X': targets.append(stim.target_x(q))
            elif pauli_type == 'Z': targets.append(stim.target_z(q))
            targets.append(stim.target_combiner())
        if targets: targets.pop()
        instructions.append(targets)
    return instructions

def compile_circuits(code, prob):
    n_qubits = code.num_qubits
    circ_z = stim.Circuit()
    circ_z.append("DEPOLARIZE1", range(n_qubits), prob)
    if hasattr(code, 'matrix_z'):
        mpp_insts = get_mpp_targets(code.matrix_z, 'Z')
        for i, targets in enumerate(mpp_insts):
            circ_z.append("MPP", targets)
            circ_z.append("DETECTOR", [stim.target_rec(-1)], [i])
    try:
        lz = code.get_logical_ops(Pauli.Z)
        mpp_insts = get_mpp_targets(lz, 'Z')
        for i, targets in enumerate(mpp_insts):
            circ_z.append("MPP", targets)
            circ_z.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], i)
    except: pass

    circ_x = stim.Circuit()
    circ_x.append("RX", range(n_qubits))
    circ_x.append("DEPOLARIZE1", range(n_qubits), prob)
    if hasattr(code, 'matrix_x'):
        mpp_insts = get_mpp_targets(code.matrix_x, 'X')
        for i, targets in enumerate(mpp_insts):
            circ_x.append("MPP", targets)
            circ_x.append("DETECTOR", [stim.target_rec(-1)], [i]) 
    try:
        lx = code.get_logical_ops(Pauli.X)
        mpp_insts = get_mpp_targets(lx, 'X')
        for i, targets in enumerate(mpp_insts):
            circ_x.append("MPP", targets)
            circ_x.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], i)
    except: pass

    sampler_z = circ_z.compile_detector_sampler()
    sampler_x = circ_x.compile_detector_sampler()
    return sampler_z, sampler_x

def generate_data_on_the_fly(sampler_z, sampler_x, num_shots):
    dets_z, obs_z = sampler_z.sample(shots=num_shots, separate_observables=True)
    dets_x, obs_x = sampler_x.sample(shots=num_shots, separate_observables=True)
    full_detectors = np.concatenate([dets_z, dets_x], axis=1).astype(np.float32)
    full_observables = np.concatenate([obs_z, obs_x], axis=1).astype(np.float32)
    return full_detectors, full_observables

# ==========================================
# 2. ユーティリティ
# ==========================================
def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def calculate_accuracy(pred_x0, target_x0):
    # -1 ~ 1 の範囲になっている pred_x0 を 0/1 に閾値判定
    pred_labels = (pred_x0.squeeze(-1) > 0.0).float()
    correct = (pred_labels == target_x0).float()
    return correct.mean().item()

def save_loss_plot(train_losses, valid_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    if valid_losses:
        plt.plot(epochs, valid_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss (Noise Prediction)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None, 0
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]
    if not checkpoints:
        return None, 0
    latest_epoch = 0
    latest_file = None
    for ckpt in checkpoints:
        match = re.search(r"model_epoch_(\d+).pth", ckpt)
        if match:
            ep = int(match.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                latest_file = ckpt
    if latest_file:
        return os.path.join(output_dir, latest_file), latest_epoch
    return None, 0

def validate(model, loader, diffusion, device, timesteps):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    loss_fn = nn.MSELoss() 
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x_0 = batch["global_labels"].to(device)
                label_y = batch["label_y"].to(device)
            else:
                label_y, x_0 = batch
                label_y = label_y.to(device)
                x_0 = x_0.to(device)

            x_0_norm = 2*x_0 - 1.0 # [0,1] -> [-1,1]
            label_y = 2*label_y - 1.0
            x_0_expanded = x_0_norm.unsqueeze(-1)
            
            eps = 1e-5
            t = torch.rand((x_0.shape[0],), device=device) * (1 - 2*eps) + eps
            
            # ノイズ生成
            noise = torch.randn_like(x_0_expanded)
            x_t, _ = diffusion.q_sample_continuous(x_0_expanded, t, noise)
            
            t_input = t * timesteps

            # ★ ノイズ予測
            # Validation時はSelf-Condなし(prev_x0=0)で評価するのが一般的、または予測値を使う
            # ここではシンプルに zeros を渡します
            pred_noise = model(x_t, t_input, label_y, prev_x0=torch.zeros_like(x_t))
            
            # Loss計算 (予測ノイズ vs 真のノイズ)
            loss = loss_fn(pred_noise, noise)

            # Accuracy計算: ノイズ予測値から x_0 を復元して判定
            pred_x0_derived = predict_start_from_noise(x_t, t, pred_noise, diffusion)
            acc = calculate_accuracy(pred_x0_derived, x_0)

            total_loss += loss.item()
            total_acc += acc
            
    return total_loss / len(loader), total_acc / len(loader)

# ==========================================
# 3. メイン処理
# ==========================================
def parse_args():
    ap = argparse.ArgumentParser(description="QEC Training Infinite Loop (Noise Pred)")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--results_dir", type=str, default="./result_noise_pred", help="Directory to save results") 
    ap.add_argument("--pretrained_path", type=str, default=None, help="Path to specific pretrained model")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--prob", type=float, default=None, help="Override noise probability")
    return ap.parse_args()

def load_cfg(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)
    return {}

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- パラメータ設定 ---
    L = int(cfg.get("DISTANCE", 6))
    M = L
    L = L*2
    if args.prob is not None:
        prob = args.prob
    else:
        prob = float(cfg.get("PROB", 0.1))
        
    samples_per_epoch = int(cfg.get("SAMPLES_PER_EPOCH", 16384)) 
    batch_size = int(cfg.get("BATCH_SIZE", 512))
    
    print('Samples per epoch:', samples_per_epoch)

    # --- Code & Circuit Setup ---
    print(f"--- Setting up BBCode L={L}, p={prob} ---")
    code = make_bbcode(L, M)
    sampler_z, sampler_x = compile_circuits(code, prob)
    
    # --- Validation Set ---
    print("Generating validation set...")
    val_dets, val_obs = generate_data_on_the_fly(sampler_z, sampler_x, num_shots=200000)
    val_dataset = TensorDataset(torch.from_numpy(val_dets), torch.from_numpy(val_obs))
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_syndrome = val_dets.shape[1]
    num_observables = val_obs.shape[1]
    
    # --- Model Setup ---
    timesteps = cfg.get("TIMESTEPS", 200) 
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device, schedule="cosine")
    print('timesteps', timesteps)
    
    # モデルの定義（変更なし）
    # ContinuousDiffusionMLPという名前でインポートしていますが中身はDiffusionTransformerです
    model = ContinuousDiffusionMLP(
        num_syndrome=num_syndrome,        # 引数名を修正 (input_dim -> num_syndrome等のマッピングが必要かも)
        num_observables=num_observables,
        hidden_size=256,
        # time_emb_dim=256, # DiTBlock側で固定されている場合もあるため、引数確認が必要
        dropout=0.1
    ).to(device)
    # ※ model_dit_zimage_prev.py の __init__ 引数に合わせて適宜修正してください

    criterion = nn.MSELoss()
    
    # --- Optimizer ---
    default_lr = 1e-3
    learning_rate = args.lr if args.lr is not None else default_lr
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # --- Resume Logic ---
    latest_ckpt_path, start_epoch = find_latest_checkpoint(output_dir)
    
    if args.pretrained_path:
        print(f"Loading specific pretrained model: {args.pretrained_path}")
        state_dict = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    elif latest_ckpt_path:
        print(f"Found checkpoint! Resuming from epoch {start_epoch}: {latest_ckpt_path}")
        state_dict = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6
    )
    
    max_epochs = 20000 
    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "best_model.pth")
    
    train_loss_history = []
    valid_loss_history = []
    
    print(f"--- Starting Infinite Training Loop (Noise Prediction) ---")

    try:
        for epoch in range(start_epoch, max_epochs):
            current_epoch_display = epoch + 1
            
            # データ生成
            dets, obs = generate_data_on_the_fly(sampler_z, sampler_x, samples_per_epoch)
            dataset = TensorDataset(torch.from_numpy(dets), torch.from_numpy(obs))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            
            model.train()
            total_train_loss = 0.0
            total_train_acc = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {current_epoch_display}")
            for batch_idx, (label_y, x_0) in enumerate(pbar):
                x_0 = x_0.to(device)
                label_y = label_y.to(device)

                x_0_norm = 2*x_0 - 1.0 # [-1, 1]
                label_y = 2*label_y - 1.0
                x_0_expanded = x_0_norm.unsqueeze(-1)
                
                # Continuous Time Sampling
                eps = 1e-5
                t = torch.rand((x_0.shape[0],), device=device) * (1 - 2*eps) + eps
                t_input = t * timesteps
                
                # ノイズ生成と注入
                noise = torch.randn_like(x_0_expanded)
                x_t, _ = diffusion.q_sample_continuous(x_0_expanded, t, noise)
                
                # =========================================================
                # ★ Self-Conditioning Training Logic (ノイズ予測版)
                # =========================================================
                prev_x0 = torch.zeros_like(x_0_expanded) # デフォルト (ヒントなし)

                target_rate = 0.5
                warmup_epochs = 50
                if epoch < warmup_epochs:
                    current_rate = target_rate * (epoch / warmup_epochs)
                else:
                    current_rate = target_rate
                
                # 50%で予測値を使用
                if np.random.rand() < current_rate:
                    with torch.no_grad():
                        # まずヒントなしでノイズを予測
                        pred_noise_est = model(x_t, t_input, label_y, prev_x0=prev_x0)
                        
                        # ★重要: ノイズ予測値そのものではなく、そこから推定した x_0 をヒントとして渡す
                        # これによりモデルは常に「データの推定値」を入力として受け取れる
                        prev_x0 = predict_start_from_noise(x_t, t, pred_noise_est, diffusion)
                        prev_x0 = prev_x0.detach()
                
                # 本番の予測: 入力は (x_t, t, y, prev_x0_hint) -> 出力は pred_noise
                pred_noise = model(x_t, t_input, label_y, prev_x0=prev_x0)
                
                # ★ Loss計算: 予測ノイズ vs 真のノイズ
                loss = criterion(pred_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                
                # Accuracy表示用: ノイズ予測から x_0 を復元して計算
                with torch.no_grad():
                    pred_x0_derived = predict_start_from_noise(x_t, t, pred_noise, diffusion)
                    acc = calculate_accuracy(pred_x0_derived, x_0)
                total_train_acc += acc
                
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)
            
            # --- Validation ---
            avg_valid_loss, avg_valid_acc = validate(model, valid_loader, diffusion, device, timesteps)
            valid_loss_history.append(avg_valid_loss)
            
            scheduler.step(avg_valid_loss)
            
            print(f"Epoch {current_epoch_display} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {avg_valid_acc:.2%}")

            # --- 保存ロジック ---
            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                torch.save(model.state_dict(), best_model_path)
            
            if current_epoch_display % 50 == 0:
                periodic_save_path = os.path.join(output_dir, f"model_epoch_{current_epoch_display}.pth")
                torch.save(model.state_dict(), periodic_save_path)
                print(f"Saved periodic checkpoint: {periodic_save_path}")
            
            if current_epoch_display % 500 == 0:
                loss_log_path = os.path.join(output_dir, f"loss_history_epoch_{current_epoch_display}.npz")
                np.savez(loss_log_path, train_loss=train_loss_history, valid_loss=valid_loss_history)
            
            save_loss_plot(train_loss_history, valid_loss_history, os.path.join(output_dir, "loss_history_current.png"))

    except KeyboardInterrupt:
        print("Training manually interrupted.")
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_interrupted_epoch_{current_epoch_display}.pth"))

    print("Training loop finished.")
    cleanup_gpu()

if __name__ == "__main__":
    main()