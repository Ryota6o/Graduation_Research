import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, device="cuda", schedule="cosine"):
        super().__init__()
        self.timesteps = timesteps
        self.device = device
        self.schedule = schedule

        # ====================================================
        # 1. Beta Schedule (離散用・互換性のため維持)
        # ====================================================
        if schedule == "linear":
            # ★ 修正箇所: T<20 でも破綻しない "Continuous Linear" スケジュール
            # 元の論文の設定 (beta: 0.1/T -> 20/T) の積分形を使用
            # alpha_bar(s) = exp( - (0.1s + 9.95s^2) )
            
            steps = timesteps + 1
            # 0.0 ~ 1.0 の進行度 s を作成
            t = torch.linspace(0, timesteps, steps) / timesteps
            
            # 積分式から alpha_bar を計算
            alphas_cumprod = torch.exp(- (0.1 * t + 9.95 * t**2))
            
            # t=0 (s=0) のときは 1.0 になるよう正規化 (exp(0)=1なので実は不要だが安全策)
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            
            # alpha_bar から beta を逆算
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            
            # 数値安定化のためクリップ (最大でも0.9999)
            self.betas = torch.clip(betas, 0.0, 0.9999).to(device)
        
        elif schedule == "cosine":
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos((t / timesteps) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # 正規化
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0, 0.9999).to(device)

        # 事前計算パラメータ (離散用)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]]
        )

    # =========================================================
    # ★ 追加: 連続時間学習・推論用メソッド
    # =========================================================
    def get_alpha_bar(self, t):
        """
        連続時間 t (0.0 ~ 1.0) を受け取り、累積信号率 alpha_bar を返す
        Schedule: Cosine
        """
        # t: [Batch] or scalar
        # alpha_bar = cos^2( t * pi / 2 )
        # コサインスケジュール専用の実装
        if self.schedule == "cosine":
            # コサインスケジュール
            return torch.cos(t * math.pi * 0.5) ** 2
            
        elif self.schedule == "linear":
            # Linearスケジュール (Continuous版)
            # alpha_bar = exp(-integral(beta))
            # beta(t) = 0.1 + (20-0.1)*t  (t: 0->1) の積分より
            # 近似式: exp( - (0.1t + 9.95t^2) )
            return torch.exp(- (0.1 * t + 9.95 * t**2))

    def q_sample_continuous(self, x_start, t, noise=None):
        """
        連続時間拡散過程
        x_start: [Batch, ...]
        t: [Batch] (0.0 ~ 1.0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # alpha_bar を計算
        alpha_bar = self.get_alpha_bar(t).to(self.device)
        
        # 形状合わせ: (Batch,) -> (Batch, 1, 1, ...)
        shape = [x_start.shape[0]] + [1] * (x_start.ndim - 1)
        alpha_bar = alpha_bar.view(*shape)
        
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar)

        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        return x_t, noise