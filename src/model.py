import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)


class Model(nn.Module):
    """CNN + GRU policy model with player-centric cross-attention pooling."""

    def __init__(self):
        super().__init__()
        self.hidden_dim: int = CONFIG["model"]["hiddenDim"]
        self.attn_dim: int = CONFIG["model"]["attnDim"]
        self.num_heads: int = CONFIG["model"]["numHeads"]
        self.action_dim: int = CONFIG["model"]["actionDim"]
        dropout_p: float = CONFIG["model"]["dropout"]

        # 3-layer CNN backbone with CoordConv on first layer
        self.conv1 = nn.Conv2d(3 + 2, 32, kernel_size=5, stride=4)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5, stride=4)
        cnn_out_channels = 64
        self.conv3 = nn.Conv2d(128, cnn_out_channels, kernel_size=3, stride=2)

        # Player-centric cross-attention pooling
        self.plr_selector = nn.Conv2d(cnn_out_channels, 1, kernel_size=1, bias=False)
        self.q_proj = nn.Linear(cnn_out_channels, self.attn_dim * self.num_heads, bias=False)
        self.k_proj = nn.Linear(cnn_out_channels, self.attn_dim * self.num_heads, bias=False)
        self.v_proj = nn.Linear(cnn_out_channels, self.attn_dim * self.num_heads, bias=False)
        self.plr_proj = nn.Linear(cnn_out_channels, self.hidden_dim)
        self.out_proj = nn.Linear(self.num_heads * self.attn_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Per-head learnable logit scale initialized to 1/0.1 = 10.0 (log scale = 2.3026)
        self.logit_scale = nn.Parameter(torch.ones(1, self.num_heads, 1, 1) * math.log(1.0 / 0.1))

        # Temporal processing and output
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.policy_head = nn.Linear(self.hidden_dim, self.action_dim)

        self.init_params()

    def init_params(self):
        """Initialize weights with He/Kaiming for Conv and Xavier for GRU/Attention."""
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self.plr_selector.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.kaiming_normal_(self.plr_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.plr_proj.bias)
        nn.init.kaiming_normal_(self.out_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.out_proj.bias)

        nn.init.xavier_uniform_(self.gru.weight_ih_l0)
        nn.init.xavier_uniform_(self.gru.weight_hh_l0)
        nn.init.zeros_(self.gru.bias_ih_l0)
        nn.init.zeros_(self.gru.bias_hh_l0)

        nn.init.xavier_uniform_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def conv_forward(self, X: Tensor) -> Tensor:
        """Applies CoordConv and sequential conv layers.

        Args:
            X: [N, C, H, W]

        Returns:
            Tensor of shape [N, C', H', W'].
        """
        batch_size, _, h, w = X.size()
        y_coords = torch.linspace(-1, 1, h, device=X.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=X.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        X = torch.cat([X, y_coords, x_coords], dim=1)

        X = torch.relu(self.conv1(X))
        X = torch.relu(self.conv2(X))
        X = torch.relu(self.conv3(X))
        return X

    def player_centric_attention(self, X_conv: Tensor) -> Tensor:
        """Pools CNN spatial features via player-centric cross-attention.

        Args:
            X_conv: [B, C_in, H_conv, W_conv] features from the CNN.

        Returns:
            Projected attention context tensor of shape [B, D].
        """
        B, C_in, _, _ = X_conv.shape

        plr_channel = self.plr_selector(X_conv)  # [B, 1, H_conv, W_conv]
        plr_channel = plr_channel.view(B, -1)  # [B, H_conv*W_conv]
        plr_weights = torch.softmax(plr_channel, dim=-1)  # [B, H_conv*W_conv]

        X_flat = X_conv.view(B, C_in, -1).transpose(1, 2)  # [B, H_conv*W_conv, C_in]
        plr_feat = torch.bmm(plr_weights.unsqueeze(1), X_flat)  # [B, 1, C_in]

        q = self.q_proj(plr_feat)  # [B, 1, H_heads * D_attn]
        k = self.k_proj(X_flat)  # [B, H_conv*W_conv, H_heads * D_attn]
        v = self.v_proj(X_flat)  # [B, H_conv*W_conv, H_heads * D_attn]

        q = q.view(B, 1, self.num_heads, self.attn_dim).transpose(1, 2)  # [B, H_heads, 1, D_attn]
        k = k.view(B, -1, self.num_heads, self.attn_dim).transpose(1, 2)  # [B, H_heads, H_conv*W_conv, D_attn]
        v = v.view(B, -1, self.num_heads, self.attn_dim).transpose(1, 2)  # [B, H_heads, H_conv*W_conv, D_attn]

        # Apply QK-normalization (Cosine Similarity Attention) to prevent softmax saturation
        q = torch.nn.functional.normalize(q, p=2, dim=-1)
        k = torch.nn.functional.normalize(k, p=2, dim=-1)

        # Scale by per-head learnable logit scale
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        scores = q @ k.transpose(-1, -2) * logit_scale  # [B, H_heads, 1, H_conv*W_conv]
        attn_probs = torch.softmax(scores, dim=-1)  # [B, H_heads, 1, H_conv*W_conv]

        context = attn_probs @ v  # [B, H_heads, 1, D_attn]
        context = context.reshape(B, -1)  # [B, H_heads * D_attn]

        # Residual skip connection
        X_proj = self.out_proj(context) + self.plr_proj(plr_feat.squeeze(1))  # [B, D]
        X_proj = torch.relu(X_proj)
        X_proj = self.ln(X_proj)
        X_proj = self.dropout(X_proj)
        return X_proj

    def forward(self, X: Tensor, prev_h: Tensor) -> tuple[Tensor, Tensor]:
        """Pass inputs through CNN + GRU + policy head.

        Args:
            X: [N, T, H, W, C]
            prev_h: [N, L, D]

        Returns:
            Logits and new hidden state of shapes [N, T, V] and [N, L, D].
        """
        N, T, H, W, C = X.shape

        X = X.view(N * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        X_conv = self.conv_forward(X)

        X_proj = self.player_centric_attention(X_conv).view(N, T, self.hidden_dim)

        gru_out, h = self.gru(X_proj, prev_h.transpose(0, 1).contiguous())  # [N, T, D]

        gru_out = gru_out + X_proj
        gru_out = self.dropout(gru_out)

        logits = self.policy_head(gru_out.reshape(N * T, -1)).view(N, T, self.action_dim)
        return logits, h.transpose(0, 1).contiguous()
