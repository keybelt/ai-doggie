import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)


class Model(nn.Module):
    """CNN + GRU policy model with learned-query cross-attention pooling and sub-block MoE policy heads."""

    def __init__(self):
        super().__init__()
        self.hidden_dim: int = CONFIG["model"]["hiddenDim"]
        self.attn_dim: int = CONFIG["model"]["attnDim"]
        self.num_heads: int = CONFIG["model"]["numHeads"]
        self.action_dim: int = CONFIG["model"]["actionDim"]

        # 3-layer CNN backbone with CoordConv on first layer
        self.conv1 = nn.Conv2d(3 + 2, 32, kernel_size=5, stride=4)
        self.gn1 = nn.GroupNorm(4, 32)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=5, stride=4)
        self.gn2 = nn.GroupNorm(8, 128)
        cnn_out_channels = 64
        self.conv3 = nn.Conv2d(128, cnn_out_channels, kernel_size=3, stride=2)
        self.gn3 = nn.GroupNorm(8, cnn_out_channels)

        # Cascaded Two-Stage Cross-Attention pooling
        attn_total_dim = self.num_heads * self.attn_dim
        self.player_query = nn.Parameter(torch.randn(1, 1, attn_total_dim) * 0.02)

        self.mha1 = nn.MultiheadAttention(
            embed_dim=attn_total_dim,
            num_heads=self.num_heads,
            kdim=cnn_out_channels,
            vdim=cnn_out_channels,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(attn_total_dim)

        self.mha2 = nn.MultiheadAttention(
            embed_dim=attn_total_dim,
            num_heads=self.num_heads,
            kdim=cnn_out_channels,
            vdim=cnn_out_channels,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(attn_total_dim)

        self.out_proj = nn.Linear(2 * attn_total_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Temporal processing and Sub-Block MoE policy heads
        self.num_policy_heads = CONFIG["model"]["numPolicyHeads"]
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.policy_heads = nn.Linear(self.hidden_dim, self.num_policy_heads * self.action_dim)

        # Pre-compute spatial CoordConv meshgrid buffers
        h, w = CONFIG["frame"]["height"], CONFIG["frame"]["width"]
        y_coords = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
        x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
        self.register_buffer("y_coords", y_coords, persistent=False)
        self.register_buffer("x_coords", x_coords, persistent=False)

    def conv_forward(self, X: Tensor) -> Tensor:
        """Applies CoordConv using cached grid buffers and sequential conv layers.

        Args:
            X: [N, C, H, W]

        Returns:
            Tensor of shape [N, C', H', W'].
        """
        batch_size, _, h, w = X.size()
        y = self.y_coords.expand(batch_size, 1, h, w)
        x = self.x_coords.expand(batch_size, 1, h, w)
        X = torch.cat([X, y, x], dim=1)

        X = F.gelu(self.gn1(self.conv1(X)))
        X = F.gelu(self.gn2(self.conv2(X)))
        X = F.gelu(self.gn3(self.conv3(X)))
        return X

    def cross_attention_pooling(self, X_conv: Tensor) -> Tensor:
        """Pools CNN spatial features via two-stage cascaded cross-attention.

        Args:
            X_conv: [B, C_in, H_conv, W_conv] features from the CNN.

        Returns:
            Projected attention context tensor of shape [B, D].
        """
        B, C_in, _, _ = X_conv.shape

        X_flat = X_conv.view(B, C_in, -1).transpose(1, 2)  # [B, H_conv*W_conv, C_in]

        # Stage 1: Extract Player
        q0 = self.player_query.expand(B, -1, -1)  # [B, 1, attn_total_dim]
        z1, _ = self.mha1(query=q0, key=X_flat, value=X_flat, need_weights=False)
        z1 = self.ln1(z1.squeeze(1))  # [B, attn_total_dim]

        # Stage 2: Query Hazards conditioned on dynamic Player State
        q1 = z1.unsqueeze(1)  # [B, 1, attn_total_dim]
        z2, _ = self.mha2(query=q1, key=X_flat, value=X_flat, need_weights=False)
        z2 = self.ln2(z2.squeeze(1))  # [B, attn_total_dim]

        # Concatenate player state and trajectory hazard context
        combined = torch.cat([z1, z2], dim=-1)  # [B, 2 * attn_total_dim]

        # Linear Projection + LayerNorm into GRU
        X_proj = self.ln(self.out_proj(combined))
        return X_proj

    def forward(self, X: Tensor, prev_h: Tensor) -> tuple[Tensor, Tensor]:
        """Pass inputs through CNN + GRU + Sub-Block MoE policy heads.

        Args:
            X: [N, T, H, W, C]
            prev_h: [N, L, D]

        Returns:
            Logits and new hidden state of shapes [K, N, T, 2] and [N, L, D].
        """
        N, T, H, W, C = X.shape

        X = X.view(N * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        X_conv = self.conv_forward(X)

        X_proj = self.cross_attention_pooling(X_conv).view(N, T, self.hidden_dim)

        gru_out, h = self.gru(X_proj, prev_h.transpose(0, 1).contiguous())  # [N, T, D]

        logits = self.policy_heads(gru_out).view(N, T, self.num_policy_heads, self.action_dim).permute(2, 0, 1, 3)
        return logits, h.transpose(0, 1).contiguous()
