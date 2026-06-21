import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    def __init__(self, frequency_embedding_size=256, max_period = 1000):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    def forward(self, t):
        dim = self.frequency_embedding_size
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

class ResBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
    def forward(self, x):
        return x + self.ff(self.norm(x))

class ContinuousDiffusionMLP(nn.Module):
    def __init__(
        self, 
        input_dim, 
        num_observables, 
        num_detectors, 
        hidden_size=1024, 
        time_emb_dim=256,
        dropout_rate=0.1
    ):
        super().__init__()
        
        # Self-Conditioning用にチャネル数が2倍になる (x_t と prev_x0 を結合するため)
        # input_dim=1 なら、モデルへの入力は 2 になる
        self.input_dim = input_dim
        self.num_observables = num_observables
        self.flat_input_dim = num_observables * input_dim
        
        # 入力層: x_t + prev_x0 なので次元は2倍
        self.concat_input_dim = self.flat_input_dim * 2 
        
        self.y_dim = time_emb_dim + num_detectors
        
        self.t_embedder = TimestepEmbedder(time_emb_dim)
        
        # 入力プロジェクション
        self.input_proj = nn.Sequential(
            nn.Linear(self.concat_input_dim + self.y_dim, hidden_size),
            nn.SiLU()
        )
        
        self.res_block1 = ResBlock(hidden_size, dropout_rate)
        
        self.mid_proj = nn.Sequential(
            nn.Linear(hidden_size + self.y_dim, hidden_size),
            nn.SiLU()
        )

        self.res_block2 = ResBlock(hidden_size, dropout_rate)
        
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.flat_input_dim)

    def forward(self, x, t, label_y, prev_x0=None):
        """
        x: [Batch, num_observables, 1] (Noisy data x_t)
        t: [Batch] (Time steps)
        label_y: [Batch, num_detectors] (Syndrome)
        prev_x0: [Batch, num_observables, 1] (Self-Conditioning input)
                 Noneの場合はゼロで埋める（推論の初回や学習時の確率的なドロップ用）
        """
        batch_size = x.shape[0]

        # 1. Self-Conditioning Input Handling
        if prev_x0 is None:
            prev_x0 = torch.zeros_like(x)
        
        # x_t と prev_x0 を結合 -> [Batch, num_observables, 2]
        x_combined = torch.cat([x, prev_x0], dim=-1)
        
        # Flatten -> [Batch, num_observables * 2]
        l_in = x_combined.view(batch_size, -1)

        # 2. Time & Condition Embedding
        t_emb = self.t_embedder(t)
        # y = cat(time, syndrome)
        y = torch.cat([t_emb, label_y], dim=1)

        # 3. Main Network
        # 入力と条件を結合
        h_in = torch.cat([l_in, y], dim=1)
        
        h = self.input_proj(h_in)
        h = self.res_block1(h)
        
        # 中間で再度条件を注入 (Skip connection like structure)
        h_cat = torch.cat([h, y], dim=1)
        h = h + self.mid_proj(h_cat) # Residual
        
        h = self.res_block2(h)
        h = self.final_norm(h)
        
        out = self.output_layer(h)
        
        # 出力形状を元に戻す [Batch, num_observables, 1]
        return out.view(batch_size, self.num_observables, self.input_dim)