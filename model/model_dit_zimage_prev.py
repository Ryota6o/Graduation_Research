# model/model_dit_zimage.py
import torch
import torch.nn as nn
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=1000): # max_period=1000に変更
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period # インスタンス変数として保持

    def timestep_embedding(self, t, dim): # staticmethodをやめてselfを使う
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# DiTBlock は変更なし
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mode="in_context", mlp_ratio=4.0, dropout=0.1, enable_modulation=True):
        super().__init__()
        self.mode = mode
        self.enable_modulation = enable_modulation
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=not enable_modulation, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        if self.mode == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=not enable_modulation, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=not enable_modulation, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )
        if self.enable_modulation:
            if self.mode == "cross_attn":
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 9 * hidden_size, bias=True) 
                )
            else:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, 6 * hidden_size, bias=True) 
                )
        else:
            self.adaLN_modulation = None

    def forward(self, x, c=None, syndrome_emb=None):
        shift_msa = scale_msa = gate_msa = None
        shift_mlp = scale_mlp = gate_mlp = None
        shift_cross = scale_cross = gate_cross = None

        if self.enable_modulation and c is not None:
            if self.mode == "cross_attn":
                (shift_msa, scale_msa, gate_msa, 
                 shift_cross, scale_cross, gate_cross, 
                 shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(9, dim=1)
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        x_norm1 = self.norm1(x)
        if self.enable_modulation:
            x_mod1 = modulate(x_norm1, shift_msa, scale_msa)
        else:
            x_mod1 = x_norm1
            
        attn_out, _ = self.attn(x_mod1, x_mod1, x_mod1)
        
        if self.enable_modulation:
            x = x + gate_msa.unsqueeze(1) * attn_out
        else:
            x = x + attn_out

        if self.mode == "cross_attn":
            x_norm_c = self.norm_cross(x)
            if self.enable_modulation:
                x_mod_c = modulate(x_norm_c, shift_cross, scale_cross)
            else:
                x_mod_c = x_norm_c
            cross_out, _ = self.cross_attn(x_mod_c, syndrome_emb, syndrome_emb)
            if self.enable_modulation:
                x = x + gate_cross.unsqueeze(1) * cross_out
            else:
                x = x + cross_out

        x_norm2 = self.norm2(x)
        if self.enable_modulation:
            x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)
        else:
            x_mod2 = x_norm2
        mlp_out = self.mlp(x_mod2)
        if self.enable_modulation:
            x = x + gate_mlp.unsqueeze(1) * mlp_out
        else:
            x = x + mlp_out

        return x

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_observables=24,
        num_syndrome=72,
        hidden_size=256,
        depth=6,
        num_heads=4,
        mode="in_context", 
        dropout=0.1,
        n_refiner_layers=2
    ):
        super().__init__()
        self.mode = mode
        self.num_observables = num_observables
        self.num_syndrome = num_syndrome

        # Embedding Layers
        # ★ Self-Conditioning対応: xの次元が 1 -> 2 に増える (x_t と prev_x0 を結合)
        self.x_embedder = nn.Linear(2, hidden_size) 
        self.syndrome_embedder = nn.Linear(1, hidden_size)
        
        # max_period=1000 に設定
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=1000)
        
        self.x_pos_embed = nn.Parameter(torch.zeros(1, num_observables, hidden_size))
        self.y_pos_embed = nn.Parameter(torch.zeros(1, num_syndrome, hidden_size))

        self.noise_refiner = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, mode=None,
                dropout=dropout, enable_modulation=True
            ) for _ in range(n_refiner_layers)
        ])

        self.context_refiner = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, mode=None, 
                dropout=dropout, enable_modulation=False
            ) for _ in range(n_refiner_layers)
        ])

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, num_heads, mode=mode, 
                dropout=dropout, enable_modulation=True
            ) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.final_linear = nn.Linear(hidden_size, 1)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_weights)

        def _zero_init(m):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)
        
        for block in self.blocks:
            if block.adaLN_modulation is not None:
                _zero_init(block.adaLN_modulation[-1])
        for block in self.noise_refiner:
            if block.adaLN_modulation is not None:
                _zero_init(block.adaLN_modulation[-1])
                
        _zero_init(self.final_adaLN[-1])

    # ★ 引数に prev_x0 を追加
    def forward(self, x, t, label_y, prev_x0=None):
        if x.dim() == 2: x = x.unsqueeze(-1)
        if label_y.dim() == 2: label_y = label_y.unsqueeze(-1)
        
        # ★ Self-Conditioning: x_t と prev_x0 を結合
        if prev_x0 is None:
            prev_x0 = torch.zeros_like(x)
        
        # x: [Batch, Obs, 1], prev_x0: [Batch, Obs, 1] -> [Batch, Obs, 2]
        x_in = torch.cat([x, prev_x0], dim=-1)

        x_emb = self.x_embedder(x_in)
        syndrome_emb = self.syndrome_embedder(label_y)
        t_emb = self.t_embedder(t)

        x_emb = x_emb + self.x_pos_embed[:, :x_emb.shape[1], :]
        syndrome_emb = syndrome_emb + self.y_pos_embed[:, :syndrome_emb.shape[1], :]

        for block in self.noise_refiner:
            x_emb = block(x_emb, c=t_emb)
            
        for block in self.context_refiner:
            syndrome_emb = block(syndrome_emb, c=None)

        if self.mode == "in_context":
            h = torch.cat([syndrome_emb, x_emb], dim=1)
            syndrome_context = None 
        else:
            h = x_emb
            syndrome_context = syndrome_emb

        for block in self.blocks:
            h = block(h, c=t_emb, syndrome_emb=syndrome_context)

        if self.mode == "in_context":
            h = h[:, self.num_syndrome:, :]
        
        shift, scale = self.final_adaLN(t_emb).chunk(2, dim=1)
        h = self.final_norm(h)
        h = modulate(h, shift, scale)
        output = self.final_linear(h) 

        return output