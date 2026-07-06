"""Contains the custom CNN + GRU model.

Example:
    >>> model = Model()
    >>> model.load_state_dict(...)
"""

import json
import sys
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.append(str(Path(__file__).resolve().parent))


class Model(nn.Module):
    """CNN + GRU policy model with player-centric cross-attention pooling."""

    def __init__(self):
        super().__init__()
        with (Path(__file__).resolve().parent / "config.json").open() as f:
            config = json.load(f)
        self._hidden_dim = config["model"]["hiddenDim"]
        self._action_dim = 8
        self._attn_dim = 16
        self._num_heads = 8

        # 3-layer CNN backbone with CoordConv on first layer.
        self._conv1 = nn.Conv2d(3 + 2, 32, kernel_size=5, stride=4)
        self._conv2 = nn.Conv2d(32, 128, kernel_size=5, stride=4)
        self._conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=2)

        # Player-centric cross-attention pooling.
        in_channels = 64
        self._plr_selector = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self._q_proj = nn.Linear(in_channels, self._attn_dim * self._num_heads, bias=False)
        self._k_proj = nn.Linear(in_channels, self._attn_dim * self._num_heads, bias=False)
        self._out_proj = nn.Linear((self._num_heads + 1) * in_channels, self._hidden_dim)

        # Temporal processing and output.
        self._gru = nn.GRU(self._hidden_dim, self._hidden_dim, batch_first=True)
        self._policy_head = nn.Linear(self._hidden_dim, self._action_dim)

        self._init_params()

    def _init_params(self):
        """He/Kaiming init for conv+ReLU, Xavier for GRU, default for output."""
        for conv in [self._conv1, self._conv2, self._conv3]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(conv.bias)

        nn.init.xavier_uniform_(self._plr_selector.weight)
        nn.init.xavier_uniform_(self._q_proj.weight)
        nn.init.xavier_uniform_(self._k_proj.weight)
        nn.init.kaiming_normal_(self._out_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self._out_proj.bias)

        nn.init.xavier_uniform_(self._gru.weight_ih_l0)
        nn.init.xavier_uniform_(self._gru.weight_hh_l0)
        nn.init.zeros_(self._gru.bias_ih_l0)
        nn.init.zeros_(self._gru.bias_hh_l0)

        nn.init.xavier_uniform_(self._policy_head.weight)
        nn.init.zeros_(self._policy_head.bias)

    def _conv_forward(
        self,
        X: Tensor,
    ) -> Tensor:
        """Sequential conv layers with CoordConv, ReLU, and stride-only downsampling.

        Args:
            X: [N, C, H, W].

        Returns:
            Tensor of shape [N, C', H', W'].
        """
        batch_size, _, h, w = X.size()
        y_coords = torch.linspace(-1, 1, h, device=X.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_coords = torch.linspace(-1, 1, w, device=X.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        X = torch.cat([X, y_coords, x_coords], dim=1)

        X = torch.relu(self._conv1(X))
        X = torch.relu(self._conv2(X))
        X = torch.relu(self._conv3(X))
        return X

    def forward(
        self,
        X: Tensor,
        prev_h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Pass inputs through CNN + GRU + policy head.

        Args:
            X: [N, T, H, W, C].
            prev_h: [N, L, D].

        Returns:
            Logits and the new hidden state of shapes [N, T, V] and [N, L, D].
        """
        N, T, H, W, C = X.shape

        # Convolve frame with combined batch size and time since convolution isn't sequential.
        # Permute from NHWC to NCHW for nn.Conv2d.
        X = X.view(N * T, H, W, C).permute(0, 3, 1, 2).contiguous()
        X_conv = self._conv_forward(X)

        B, C_in, _, _ = X_conv.shape

        plr_channel = self._plr_selector(X_conv)  # [B, 1, H_conv, W_conv]
        plr_channel = plr_channel.view(B, -1)  # [B, H_conv*W_conv]
        plr_weights = torch.softmax(plr_channel, dim=-1)  # [B, H_conv*W_conv]

        X_flat = X_conv.view(B, C_in, -1).transpose(1, 2)  # [B, H_conv*W_conv, C_in]
        plr_feat = torch.bmm(plr_weights.unsqueeze(1), X_flat)  # [B, 1, C_in]

        q = self._q_proj(plr_feat)  # [B, 1, H_heads * D_attn]
        k = self._k_proj(X_flat)  # [B, H_conv*W_conv, H_heads * D_attn]

        q = q.view(B, 1, self._num_heads, self._attn_dim).transpose(1, 2)  # [B, H_heads, 1, D_attn]
        k = k.view(B, -1, self._num_heads, self._attn_dim).transpose(1, 2)  # [B, H_heads, H_conv*W_conv, D_attn]

        scores = torch.matmul(q, k.transpose(-1, -2)) * self._attn_dim**-0.5  # [B, H_heads, 1, H_conv*W_conv]
        attn_probs = torch.softmax(scores, dim=-1)  # [B, H_heads, 1, H_conv*W_conv]

        X_flat_with_head = X_flat.unsqueeze(1).expand(-1, self._num_heads, -1, -1)  # [B, H_heads, H_conv*W_conv, C_in]
        context = attn_probs @ X_flat_with_head  # [B, H_heads, 1, C_in]
        context = context.reshape(B, -1)  # [B, H_heads * C_in]

        combined = torch.cat([plr_feat.squeeze(1), context], dim=-1)  # [B, (H_heads + 1) * C_in]

        X_proj = torch.relu(self._out_proj(combined))  # [B, D]
        X_proj = X_proj.view(N, T, self._hidden_dim)

        gru_out, h = self._gru(
            X_proj,
            # nn.GRU expects (L, N, D).
            prev_h.transpose(0, 1).contiguous(),
        )  # [N, T, D]

        gru_out = gru_out.reshape(N * T, -1)
        logits_nonsequential = self._policy_head(gru_out)
        logits = logits_nonsequential.view(N, T, self._action_dim)

        return logits, h.transpose(0, 1).contiguous()
