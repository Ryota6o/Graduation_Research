'''エポック毎にStimから新しいデータを取得 + 高速化対応版 (AMP, DataLoader並列化, torch.compile)'''
'''修正点: StimIterableDatasetのyield順序を修正 (obs, dets -> dets, obs)'''

import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np
from time import time
import gc
import yaml
import math

# --- 必要なライブラリのインポート (Stim, QLDPC) ---
import stim
import sympy
from qldpc import codes
from qldpc.objects import Pauli

# --- モデル定義のインポート ---
from model.model_ffn_gemini import ContinuousDiffusionMLP 
from diffusion.diffusion import GaussianDiffusion

# ==========================================
# 0. 高速化設定
# ==========================================
torch.backends.cudnn.benchmark = True 
torch.set_float32_matmul_precision('high') 

# ==========================================
# 1. データ生成ロジック & Dataset
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

def create_circuits(code, prob):
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

    return circ_z, circ_x

# ★★★ 修正箇所: yieldの順序を (obs, dets) から (dets, obs) に変更 ★★★
class StimIterableDataset(IterableDataset):
    def __init__(self, circ_z, circ_x, total_samples, batch_size_gen=1024):
        self.circ_z = circ_z
        self.circ_x = circ_x
        self.total_samples = total_samples
        self.batch_size_gen = batch_size_gen 

    def __iter__(self):
        # ワーカー内でコンパイル
        sampler_z = self.circ_z.compile_detector_sampler()
        sampler_x = self.circ_x.compile_detector_sampler()

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = self.total_samples
        else:
            per_worker = int(math.ceil(self.total_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.total_samples)

        num_needed = iter_end - iter_start
        current_count = 0
        
        while current_count < num_needed:
            fetch_size = min(self.batch_size_gen, num_needed - current_count)
            
            dets_z, obs_z = sampler_z.sample(shots=fetch_size, separate_observables=True)
            dets_x, obs_x = sampler_x.sample(shots=fetch_size, separate_observables=True)
            
            batch_dets = np.concatenate([dets_z, dets_x], axis=1).astype(np.float32)
            batch_obs = np.concatenate([obs_z, obs_x], axis=1).astype(np.float32)
            
            for i in range(fetch_size):
                # ★ ここを修正: dets(syndrome) が先、obs(logical) が後
                yield torch.from_numpy(batch_dets[i]), torch.from_numpy(batch_obs[i])
            
            current_count += fetch_size

def generate_data_once(sampler_z, sampler_x, num_shots):
    dets_z, obs_z = sampler_z.sample(shots=num_shots, separate_observables=True)
    dets_x, obs_x = sampler_x.sample(shots=num_shots, separate_observables=True)
    full_detectors = np.concatenate([dets_z, dets_x], axis=1).astype(np.float32)
    full_observables = np.concatenate([obs_z, obs_x], axis=1).astype(np.float32)
    return full_detectors, full_observables

# ==========================================
# 2. ユーティリティ
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        save_model = model.module if hasattr(model, "module") else model
        save_model = save_model._orig_mod if hasattr(save_model, "_orig_mod") else save_model
        torch.save(save_model.state_dict(), self.path)
        self.val_loss_min = val_loss

def calculate_accuracy(pred_x0, target_x0):
    pred_labels = (pred_x0.squeeze(-1) > 0.0).float()
    correct = (pred_labels == target_x0).float()
    return correct.mean().item()

def save_loss_plot(train_losses, valid_losses, save_path):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    if valid_losses:
        plt.plot(epochs, valid_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def validate(model, loader, diffusion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    loss_fn = nn.MSELoss() 
    
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x_0 = batch["global_labels"].to(device, non_blocking=True)
                label_y = batch["label_y"].to(device, non_blocking=True)
            else:
                label_y, x_0 = batch
                label_y = label_y.to(device, non_blocking=True)
                x_0 = x_0.to(device, non_blocking=True)

            x_0_norm = x_0 - 0.5
            label_y = label_y - 0.5
            x_0_expanded = x_0_norm.unsqueeze(-1)
            
            t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device).long()
            noise = torch.randn_like(x_0_expanded)
            x_t, _ = diffusion.q_sample(x_0_expanded, t, noise)

            with torch.amp.autocast('cuda',dtype = torch.bfloat16):
                pred_x0 = model(x_t, t, label_y)
                loss = loss_fn(pred_x0.squeeze(-1), x_0_norm)
            
            acc = calculate_accuracy(pred_x0.float(), x_0.float())
            total_loss += loss.item()
            total_acc += acc
            
    return total_loss / len(loader), total_acc / len(loader)

# ==========================================
# 3. メイン処理
# ==========================================
def parse_args():
    ap = argparse.ArgumentParser(description="QEC Training Optimized")
    ap.add_argument("--config", default="config.yaml", help="path to config.yaml")
    ap.add_argument("--results_dir", type=str, default="./result_opt", help="Directory to save results") 
    ap.add_argument("--pretrained_path", type=str, default=None, help="Path to pretrained .pth")
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

    L = int(cfg.get("DISTANCE", 6))
    #M = L
    M = L
    #L = L*2
    if args.prob is not None:
        prob = args.prob
    else:
        prob = float(cfg.get("PROB", 0.1))
        
    samples_per_epoch = int(cfg.get("SAMPLES_PER_EPOCH", 16384)) 
    batch_size = int(cfg.get("BATCH_SIZE", 512))
    
    print('1エポックにつきのサンプル数', samples_per_epoch)
    print(f"--- Setting up BBCode L={L}, p={prob} ---")
    code = make_bbcode(L, M)
    
    print("Compiling circuits...")
    circ_z, circ_x = create_circuits(code, prob)
    
    sampler_z = circ_z.compile_detector_sampler()
    sampler_x = circ_x.compile_detector_sampler()
    
    print("Generating validation set...")
    val_dets, val_obs = generate_data_once(sampler_z, sampler_x, num_shots=200000)
    
    from torch.utils.data import TensorDataset
    val_dataset = TensorDataset(torch.from_numpy(val_dets), torch.from_numpy(val_obs))
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                              num_workers=2, pin_memory=True)

    num_syndrome = val_dets.shape[1]
    num_observables = val_obs.shape[1]
    print(f"Validation Data Shape -> Syndrome: {num_syndrome}, Observables: {num_observables}")
    
    timesteps = cfg.get("TIMESTEPS", 200) 
    print('拡散時刻は', timesteps)
    diffusion = GaussianDiffusion(timesteps=timesteps, device=device, schedule="cosine")

    model = ContinuousDiffusionMLP(
        input_dim=1,
        num_observables=num_observables,
        num_detectors=num_syndrome,
        hidden_size=2048,
        time_emb_dim=256,
        dropout_rate=0.1
    ).to(device)

    if args.pretrained_path:
        if os.path.exists(args.pretrained_path):
            print(f"Loading weights: {args.pretrained_path}")
            state_dict = torch.load(args.pretrained_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("_orig_mod.", "").replace("module.", "")
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict)
                print("Weights loaded successfully.")
            except RuntimeError as e:
                print(f"Error loading weights: {e}")
                return

    try:
        print("Compiling model with torch.compile...")
        # 安定動作のため default モード
        model = torch.compile(model, mode="default")
    except Exception as e:
        print(f"Warning: torch.compile failed. Using standard mode. Error: {e}")

    criterion = nn.MSELoss()
    default_lr = 1e-3
    learning_rate = args.lr if args.lr is not None else default_lr
    min_lr = 1e-5
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')

    epochs = cfg.get("EPOCHS", 100)
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    #target_decay_steps = 2000 
   
    target_decay_steps = steps_per_epoch * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=target_decay_steps, eta_min=min_lr
    )
    
    #early_stopping = EarlyStopping(patience=15, verbose=True, path=os.path.join(output_dir, "best_model.pth"))

    train_loss_history = []
    valid_loss_history = []
    
    start_time = time()
    global_step = 0

    try:
        for epoch in range(epochs):
            # Dataset
            train_dataset = StimIterableDataset(circ_z, circ_x, samples_per_epoch)
            
            # DataLoader
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                num_workers=4,        
                pin_memory=True,      
                prefetch_factor=2,    
                persistent_workers=False 
            )
            
            model.train()
            total_train_loss = 0.0
            total_train_acc = 0.0
            
            pbar = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (label_y, x_0) in enumerate(pbar):
                # label_y = Syndrome, x_0 = Logical
                x_0 = x_0.to(device, non_blocking=True)
                label_y = label_y.to(device, non_blocking=True)

                x_0_norm = x_0 - 0.5
                label_y = label_y - 0.5
                x_0_expanded = x_0_norm.unsqueeze(-1)
                
                t = torch.randint(0, diffusion.timesteps, (x_0.shape[0],), device=device).long()
                noise = torch.randn_like(x_0_expanded)
                x_t, _ = diffusion.q_sample(x_0_expanded, t, noise)
                
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast('cuda', dtype = torch.bfloat16):
                    pred_x0 = model(x_t, t, label_y)
                    loss = criterion(pred_x0.squeeze(-1), x_0_norm)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                #if global_step < target_decay_steps:
                scheduler.step()
                global_step += 1

                total_train_loss += loss.item()
                with torch.no_grad():
                    acc = calculate_accuracy(pred_x0.float(), x_0.float())
                total_train_acc += acc

                pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2%}"})
            
            avg_train_loss = total_train_loss / (batch_idx + 1)
            train_loss_history.append(avg_train_loss)
            
            avg_valid_loss, avg_valid_acc = validate(model, valid_loader, diffusion, device)
            valid_loss_history.append(avg_valid_loss)
            
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {avg_valid_acc:.2%}")

            if (epoch + 1) % 50 == 0:
                save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save(save_model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))

            '''early_stopping(avg_valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break'''

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        save_loss_plot(train_loss_history, valid_loss_history, os.path.join(output_dir, "loss_history.png"))
        end_time = time()
        print(f"Total Time: {(end_time - start_time)/60:.1f} min")

if __name__ == "__main__":
    main()