'''無限学習(20000epoch) + レジューム機能 + ReduceLROnPlateau + 高速化(AMP/Prefetch) + 経過時間表示'''

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
from time import time
import gc
import yaml
import re
from concurrent.futures import ThreadPoolExecutor

# --- 必要なライブラリのインポート (Stim, QLDPC) ---
import stim
import sympy
from qldpc import codes
from qldpc.objects import Pauli

# --- モデル定義のインポート ---
from model.model_dit_zimage import DiffusionTransformer
from diffusion.diffusion import GaussianDiffusion

# ==========================================
# 0. ヘルパー関数 (時間表示用)
# ==========================================
def format_time(seconds):
    """秒数を 'Xh Ym Zs' 形式の文字列に変換"""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

# ==========================================
# 1. データ生成ロジック
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
    
    # --- Z基底 ---
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

    # --- X基底 ---
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

# --- 非同期データ生成クラス (高速化用) ---
class AsyncDataGenerator:
    def __init__(self, sampler_z, sampler_x, samples_per_epoch, batch_size):
        self.sampler_z = sampler_z
        self.sampler_x = sampler_x
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.start_next_generation()

    def start_next_generation(self):
        self.future = self.executor.submit(
            generate_data_on_the_fly, 
            self.sampler_z, 
            self.sampler_x, 
            self.samples_per_epoch
        )

    def get_data(self):
        dets, obs = self.future.result()
        self.start_next_generation()
        return dets, obs

# ==========================================
# 2. ユーティリティ・クラス
# ==========================================

class WeightedMSELoss(nn.Module):
    def __init__(self, error_weight=10.0):
        super().__init__()
        self.error_weight = error_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, input, target):
        loss = self.mse(input, target)
        weights = torch.ones_like(target)
        weights.masked_fill_(target > 0, self.error_weight)
        return (loss * weights).mean()

class BestModelSaver:
    def __init__(self, save_path='best_model.pth', verbose=True):
        self.best_loss = float('inf')
        self.save_path = save_path
        self.verbose = verbose
    
    def check_and_save(self, current_loss, model, elapsed_str=None):
        if current_loss < self.best_loss:
            if self.verbose:
                time_msg = f" [Time: {elapsed_str}]" if elapsed_str else ""
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {current_loss:.6f}). Saving...{time_msg}")
            self.best_loss = current_loss
            torch.save(model.state_dict(), self.save_path)
            return True
        return False

def cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def calculate_metrics_fast(pred_x0, target_x0):
    with torch.no_grad():
        pred_labels = (pred_x0.squeeze(-1) > 0.0).float()
        target_labels = (target_x0 > 0.0).float()
        correct = (pred_labels == target_labels).float().sum()
        total = torch.tensor(target_labels.numel(), device=target_labels.device)
        mask_error = (target_labels == 1.0)
        total_positives = mask_error.sum()
        true_positives = (pred_labels[mask_error] == 1.0).float().sum()
        return correct, total, true_positives, total_positives

def save_loss_plot(train_losses, valid_losses, train_recalls, valid_recalls, save_path):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    min_len = min(len(train_losses), len(valid_losses))
    t_loss = train_losses[:min_len]
    v_loss = valid_losses[:min_len]
    t_rec = train_recalls[:min_len]
    v_rec = valid_recalls[:min_len]
    
    epochs = range(1, min_len + 1)
    
    # Loss Plot
    ax[0].plot(epochs, t_loss, label='Training Loss', color='blue')
    ax[0].plot(epochs, v_loss, label='Validation Loss', color='orange')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].grid(True, which="both", ls="-", alpha=0.5)
    
    # Recall Plot
    ax[1].plot(epochs, t_rec, label='Training Recall', color='blue')
    ax[1].plot(epochs, v_rec, label='Validation Recall', color='orange')
    ax[1].set_title('Recall')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.savefig(save_path)
    plt.close()

def validate(model, loader, diffusion, device, criterion, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples_acc = 0
    total_tp = 0
    total_pos = 0
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x_0 = batch["global_labels"].to(device, non_blocking=True)
                label_y = batch["label_y"].to(device, non_blocking=True)
            else:
                label_y, x_0 = batch
                label_y = label_y.to(device, non_blocking=True)
                x_0 = x_0.to(device, non_blocking=True)

            x_0_norm = 2*x_0.float() - 1.0
            label_y = 2*label_y.float() - 1.0

            batch_size = x_0.shape[0]
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x_0_norm.unsqueeze(-1))
            
            x_t_val, _ = diffusion.q_sample(x_0_norm.unsqueeze(-1), t, noise)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred_x0_val = model(x_t_val, t, label_y)
                loss = criterion(pred_x0_val.squeeze(-1), x_0_norm)
            
            total_loss += loss.item()
            c, t_acc, tp, pos = calculate_metrics_fast(pred_x0_val, x_0_norm)
            total_correct += c
            total_samples_acc += t_acc
            total_tp += tp
            total_pos += pos
            
    avg_loss = total_loss / num_batches
    avg_acc = (total_correct / total_samples_acc).item()
    avg_recall = (total_tp / total_pos).item() if total_pos > 0 else 0.0
    
    return avg_loss, avg_acc, avg_recall

def find_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None, 0
    
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not checkpoints:
        return None, 0
    
    latest_epoch = 0
    latest_file = None
    
    for ckpt in checkpoints:
        match = re.search(r"checkpoint_epoch_(\d+).pth", ckpt)
        if match:
            ep = int(match.group(1))
            if ep > latest_epoch:
                latest_epoch = ep
                latest_file = ckpt
                
    if latest_file:
        return os.path.join(output_dir, latest_file), latest_epoch
    return None, 0

# ==========================================
# 3. メイン処理
# ==========================================
def parse_args():
    ap = argparse.ArgumentParser(description="QEC Infinite Training")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--results_dir", type=str, default="./result_contex" , help="Directory to save results") 
    ap.add_argument("--mode", type=str, default="in_context", help="in_context or cross_attn")
    ap.add_argument("--dropout", type=float, default=0.1)
    
    ap.add_argument("--pretrained_path", type=str, default=None, help="Force load specific model")
    ap.add_argument("--prob", type=float, default=None, help="Overwrite training error probability")
    ap.add_argument("--lr", type=float, default=None, help="Overwrite learning rate")
    
    return ap.parse_args()

def load_cfg(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)
    return {}

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        use_amp = True
        print("CUDA enabled. Using AMP and cuDNN benchmark.")
    else:
        device = torch.device("cpu")
        use_amp = False
        print("CUDA not available. Running on CPU.")
    
    if args.results_dir == "./results_contex": 
        if args.mode == "in_context":
            output_dir = "./result_contex_weighted"
        elif args.mode == "cross_attn":
            output_dir = "./result_cross"
        else:
            output_dir = args.results_dir
    else:
        output_dir = args.results_dir
    
    print('現在のモード',args.mode)

    if args.pretrained_path and "FT" not in output_dir:
        output_dir = output_dir + "_FT"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f'Results will be saved to: {output_dir}')

    # --- パラメータ設定 ---
    L = int(cfg.get("DISTANCE", 6))
    M = L
    #L = L*2
    
    print('符号距離',L,M)

    default_train_prob = 0.06
    train_prob = args.prob if args.prob is not None else default_train_prob
    val_prob = train_prob

    samples_per_epoch = int(cfg.get("SAMPLES_PER_EPOCH", 16384)) 
    batch_size = int(cfg.get("BATCH_SIZE", 512))
    
    print(f"--- Setting up BBCode L={L} ---")
    code = make_bbcode(L, M)
    
    print("Compiling circuits...")
    train_sampler_z, train_sampler_x = compile_circuits(code, train_prob)
    val_sampler_z, val_sampler_x = compile_circuits(code, val_prob)
    
    print("Initializing Async Data Generator...")
    data_gen = AsyncDataGenerator(train_sampler_z, train_sampler_x, samples_per_epoch, batch_size)

    print("Generating validation set...")
    val_dets, val_obs = generate_data_on_the_fly(val_sampler_z, val_sampler_x, num_shots=200000)
    val_dataset = TensorDataset(torch.from_numpy(val_dets), torch.from_numpy(val_obs))
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

    num_syndrome = val_dets.shape[1]
    num_observables = val_obs.shape[1]
    
    timesteps = cfg.get("TIMESTEPS", 200) 
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device, schedule='cosine')

    model = DiffusionTransformer(
        num_observables=num_observables,
        num_syndrome=num_syndrome,
        hidden_size=cfg.get("HIDDEN_SIZE", 256),
        depth=cfg.get("DEPTH", 6),
        num_heads=cfg.get("NUM_HEADS", 4),
        mode=args.mode,
        dropout=args.dropout
    ).to(device)

    start_epoch = 0
    latest_ckpt_path, latest_epoch = find_latest_checkpoint(output_dir)
    
    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Loading specific pretrained model: {args.pretrained_path}")
            state_dict = torch.load(args.pretrained_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("_orig_mod.", "")
                name = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(">>> Loaded explicitly specified weights.")
        else:
            print(f"!!! Error: Path {args.pretrained_path} does not exist.")
            return
    elif latest_ckpt_path:
        print(f"Found checkpoint! Resuming from epoch {latest_epoch}: {latest_ckpt_path}")
        state_dict = torch.load(latest_ckpt_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("_orig_mod.", "")
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        start_epoch = latest_epoch
    else:
        print("No checkpoint found. Starting from scratch.")

    try:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    except Exception as e:
        print(f"torch.compile failed (ignored): {e}")

    criterion = WeightedMSELoss(error_weight=10.0).to(device)
    
    default_lr = 1e-4
    learning_rate = args.lr if args.lr is not None else default_lr
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ★ 修正: verbose 引数を削除
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=50,
        min_lr=1e-7
    )

    train_loss_history = []
    valid_loss_history = []
    train_rec_history = []
    valid_rec_history = []
    
    log_path = os.path.join(output_dir, "training_logs.npz")
    if os.path.exists(log_path) and start_epoch > 0:
        try:
            logs = np.load(log_path)
            if len(logs['train_loss']) >= start_epoch:
                train_loss_history = logs['train_loss'][:start_epoch].tolist()
                valid_loss_history = logs['valid_loss'][:start_epoch].tolist()
                train_rec_history = logs['train_rec'][:start_epoch].tolist()
                valid_rec_history = logs['valid_rec'][:start_epoch].tolist()
                print(">>> Loaded previous training logs for continuity.")
        except Exception as e:
            print(f"Warning: Could not load previous logs: {e}")

    if valid_loss_history:
        best_loss = min(valid_loss_history)
    else:
        best_loss = float('inf')

    best_saver = BestModelSaver(save_path=os.path.join(output_dir, "best_model.pth"), verbose=True)
    best_saver.best_loss = best_loss 

    max_epochs = 20000 
    
    start_time = time()
    print(f"--- Starting Training Loop (Target: {max_epochs} Epochs) ---")

    try:
        for epoch in range(start_epoch, max_epochs):
            current_elapsed = time() - start_time
            elapsed_str = format_time(current_elapsed)

            dets, obs = data_gen.get_data()
            
            samples_with_error = np.any(obs > 0.5, axis=1) 
            batch_logical_error_rate = np.mean(samples_with_error)
            
            dataset = TensorDataset(torch.from_numpy(dets), torch.from_numpy(obs))
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

            model.train()
            
            total_train_loss = torch.tensor(0.0, device=device)
            total_correct = torch.tensor(0.0, device=device)
            total_samples_acc = torch.tensor(0.0, device=device)
            total_tp = torch.tensor(0.0, device=device)
            total_pos = torch.tensor(0.0, device=device)
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} (LER: {batch_logical_error_rate:.2%})", dynamic_ncols=True)
            
            for batch_idx, (label_y, x_0) in enumerate(pbar):
                x_0 = x_0.to(device, non_blocking=True)
                label_y = label_y.to(device, non_blocking=True)

                x_0_norm = x_0.float() - 0.5
                label_y = label_y.float() - 0.5
                x_0_expanded = x_0_norm.unsqueeze(-1)
                
                t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device).long()
                noise = torch.randn_like(x_0_expanded)
                x_t, _ = diffusion.q_sample(x_0_expanded, t, noise)
                
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred_x0 = model(x_t, t, label_y)
                    loss = criterion(pred_x0.squeeze(-1), x_0_norm)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                total_train_loss += loss.detach()
                c, t_acc, tp, pos = calculate_metrics_fast(pred_x0, x_0_norm)
                total_correct += c
                total_samples_acc += t_acc
                total_tp += tp
                total_pos += pos

                if batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{current_lr:.2e}"})
            
            avg_train_loss = (total_train_loss / len(train_loader)).item()
            avg_train_acc = (total_correct / total_samples_acc).item()
            avg_train_rec = (total_tp / total_pos).item() if total_pos > 0 else 0.0
            
            train_loss_history.append(avg_train_loss)
            train_rec_history.append(avg_train_rec)
            
            # --- Validation ---
            avg_valid_loss, avg_valid_acc, avg_valid_rec = validate(model, valid_loader, diffusion, device, criterion, use_amp)
            
            valid_loss_history.append(avg_valid_loss)
            valid_rec_history.append(avg_valid_rec)
            
            print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} / {avg_valid_loss:.4f} | "
                  f"Recall: {avg_train_rec:.2%} / {avg_valid_rec:.2%} | "
                  f"Acc: {avg_train_acc:.2%} / {avg_valid_acc:.2%}")
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_valid_loss)
            new_lr = optimizer.param_groups[0]['lr']
            
            # 手動で学習率の変化を表示
            if new_lr != old_lr:
                print(f"Epoch {epoch+1}: Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
            
            best_saver.check_and_save(avg_valid_loss, model, elapsed_str)
            
            if (epoch + 1) % 50 == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path} [Time: {elapsed_str}]")
                
            save_loss_plot(train_loss_history, valid_loss_history, 
                           train_rec_history, valid_rec_history, 
                           os.path.join(output_dir, "loss_history.png"))
            
            # 毎エポックの上書き保存 (既存)
            np.savez(os.path.join(output_dir, "training_logs.npz"),
                     train_loss=train_loss_history, valid_loss=valid_loss_history,
                     train_rec=train_rec_history, valid_rec=valid_rec_history)
            
            if (epoch + 1) % 500 == 0:
                periodic_log_path = os.path.join(output_dir, f"training_logs_epoch_{epoch+1}.npz")
                np.savez(periodic_log_path,
                         train_loss=train_loss_history, valid_loss=valid_loss_history,
                         train_rec=train_rec_history, valid_rec=valid_rec_history)
                print(f"Saved periodic logs: {periodic_log_path}")

    except KeyboardInterrupt:
        print("Training interrupted.")
        torch.save(model.state_dict(), os.path.join(output_dir, f"checkpoint_interrupted_epoch_{epoch+1}.pth"))
        
    finally:
        data_gen.executor.shutdown(wait=False)
        cleanup_gpu()

    print("Training finished.")

if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()