import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    (変更なし) 論文 Eq. (22) に準拠した Time Embedding
    """
    def __init__(self, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 1000 

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
    """
    【追加】残差接続(Residual Connection)とLayerNormを持つブロック
    構造: x -> LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout -> + x
    これにより勾配消失を防ぎ、学習を安定化させる。
    """
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # Skip Connection: 入力 x をそのまま足し合わせる
        return x + self.ff(self.norm(x))

class ContinuousDiffusionMLP(nn.Module):
    def __init__(
        self,
        input_dim=1,
        num_observables=24,
        num_detectors=72,
        hidden_size=2048,
        time_emb_dim=256,
        dropout_rate=0.1 
    ):
        super().__init__()
        self.num_observables = num_observables
        self.input_dim = input_dim
        self.num_detectors = num_detectors
        
        # Time Embedding
        self.t_embedder = TimestepEmbedder(frequency_embedding_size=time_emb_dim)
        
        self.y_dim = time_emb_dim + num_detectors
        self.flat_input_dim = num_observables * input_dim
        
        # --- 入力層 (次元圧縮・拡張) ---
        in_dim_layer1 = self.flat_input_dim + self.y_dim
        
        # 最初のプロジェクション (入力次元 -> hidden_size)：Sequentialは、一方通行のシンプルなモデルのためのクラス
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim_layer1, hidden_size),
            nn.SiLU()  # ここは単純なActivations
        )

        # --- 前半ブロック (ResBlockに変更) ---
        # 以前の layer1, layer2 を ResBlock に置き換え
        # これにより深くなっても勾配が通りやすくなる
        self.res_block1 = ResBlock(hidden_size, dropout_rate)
        
        # --- 中間層 (条件再注入) ---
        # ここで y を concat するため、一時的に次元が増える
        # hidden + y_dim -> hidden に戻す層
        self.mid_proj = nn.Sequential(
            nn.Linear(hidden_size + self.y_dim, hidden_size),
            nn.SiLU()
        )

        # --- 後半ブロック (ResBlockに変更) ---
        self.res_block2 = ResBlock(hidden_size, dropout_rate)
        
        # --- 出力層 ---
        # 最後の正規化を入れておくと出力が安定する
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.flat_input_dim)

    def forward(self, x, t, label_y):
        batch_size = x.shape[0]

        # 1. Flatten
        l_t = x.view(batch_size, -1)

        # 2. Time & Condition Embedding
        t_emb = self.t_embedder(t)
        # = 2 * label_y.float() - 1

        y = torch.cat([t_emb, label_y], dim=1)

        # 3. Input Projection
        x_in = torch.cat([l_t, y], dim=1)
        h = self.input_proj(x_in)

        # 4. ResBlock 1 (Residual Connection & LayerNorm applied internally)
        h = self.res_block1(h)

        # 5. Skip Connection (条件再注入)
        h_concat = torch.cat([h, y], dim=1)
        h = self.mid_proj(h_concat)

        # 6. ResBlock 2
        h = self.res_block2(h)

        # 7. Output
        h = self.final_norm(h)
        out = self.output_layer(h)

        return out.view(batch_size, self.num_observables, self.input_dim)