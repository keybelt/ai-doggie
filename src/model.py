import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)


class Model(nn.Module):

    def __init__(self, attn_dim: int, num_heads: int):
        super().__init__()
        self.attn_dim = attn_dim
        self.num_heads = num_heads

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

        # Pre-compute spatial CoordConv meshgrid buffers
        h, w = CONFIG["frame"]["height"], CONFIG["frame"]["width"]
        y_coords = torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w)
        x_coords = torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w)
        self.register_buffer("y_coords", y_coords, persistent=False)
        self.register_buffer("x_coords", x_coords, persistent=False)

        dynamic_cfg = CONFIG["training"]["dynamic"]
        self.seq_len: int = dynamic_cfg["seqLen"]
        self.target_offset: float = float(dynamic_cfg["targetOffset"])

        visual_dim = 2 * attn_total_dim
        aux_dim = 7
        fusion_dim = visual_dim + aux_dim + self.seq_len
        self.aux_ln = nn.LayerNorm(aux_dim)
        self.fc1 = nn.Linear(fusion_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.fc3.bias.data.fill_(self.target_offset)
        self.fc3.weight.data.zero_()

    def conv_forward(self, X: Tensor) -> Tensor:
        """
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

        return torch.cat([z1, z2], dim=-1)  # [B, 2 * attn_total_dim]

    def forward(
        self,
        X: Tensor,
        aux_state: Tensor,
        actions: Tensor,
    ) -> Tensor:
        """
        Args:
            X: [B, C, H, W] raw RGB frame I_0 at spawn.
            aux_state: [B, 7] physical state [p1_vx, p1_vy, p1_grav, p2_vx, p2_vy, p2_grav, is_holding].
            actions: [B, MAX_H] raw candidate action sequence (0.0=release, 1.0=jump, -1.0=pad).

        Returns:
            Tensor of shape [B, 1] containing predicted frames to death.
        """
        X_conv = self.conv_forward(X)  # [B, C', H', W']
        z_0 = self.cross_attention_pooling(X_conv)  # [B, D]

        fused = torch.cat([z_0, self.aux_ln(aux_state), actions], dim=-1)  # [B, D + 7 + MAX_H]
        h = F.gelu(self.fc1(fused))
        h = F.gelu(self.fc2(h))
        pred_ftd = self.fc3(h)  # [B, 1]
        return pred_ftd
