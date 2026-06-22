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

sys.path.append(str(Path(__file__).resolve().parents[1]))

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
    _CONFIG = json.load(f)

_CONFIG_MODEL = _CONFIG["model"]


class Model(nn.Module):
    """CNN + GRU policy model."""

    def __init__(self):
        """Initialize hidden dim, all weights and biases, and calculate flat size."""
        super().__init__()
        self._hidden_dim = _CONFIG_MODEL["hiddenDim"]
        self._vocab_size = _CONFIG_MODEL["vocabSize"]

        shapes = _CONFIG_MODEL["weightShapes"]

        self._conv1 = nn.Conv2d(
            shapes["conv1"]["inChannels"],
            shapes["conv1"]["outChannels"],
            kernel_size=shapes["conv1"]["kernelSize"],
            stride=shapes["conv1"]["stride"],
        )

        self._conv2 = nn.Conv2d(
            shapes["conv2"]["inChannels"],
            shapes["conv2"]["outChannels"],
            kernel_size=shapes["conv2"]["kernelSize"],
            stride=shapes["conv2"]["stride"],
        )

        self._conv3 = nn.Conv2d(
            shapes["conv3"]["inChannels"],
            shapes["conv3"]["outChannels"],
            kernel_size=shapes["conv3"]["kernelSize"],
            stride=shapes["conv3"]["stride"],
        )

        self._conv4 = nn.Conv2d(
            shapes["conv4"]["inChannels"],
            shapes["conv4"]["outChannels"],
            kernel_size=shapes["conv4"]["kernelSize"],
            stride=shapes["conv4"]["stride"],
        )

        # Dynamically calculate flattened size.
        with torch.inference_mode():
            frame_H_px = _CONFIG["capture"]["frameDims"]["pipelineHeightPx"]
            frame_W_px = _CONFIG["capture"]["frameDims"]["pipelineWidthPx"]

            dummy = torch.zeros(
                1,
                shapes["conv1"]["inChannels"],
                frame_H_px,
                frame_W_px,
            )
            flat_size = self._conv_forward(dummy).numel()

        self._fc = nn.Linear(flat_size, self._hidden_dim)

        self._gru = nn.GRU(self._hidden_dim, self._hidden_dim, batch_first=True)

        self._policy_head = nn.Linear(self._hidden_dim, self._vocab_size)

        self._init_params()

    def _init_params(self):
        """He/Kaiming init for conv+ReLU, Xavier for GRU, default for output."""
        for module in [self._conv1, self._conv2, self._conv3, self._conv4]:
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(module.bias)

        nn.init.kaiming_normal_(self._fc.weight, nonlinearity="relu")
        nn.init.zeros_(self._fc.bias)

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
        """3 conv layers with ReLU, stride-only downsampling.

        Args:
            X: [N, C, H, W].

        Returns:
            Tensor of shape [N, C', H', W'].
        """
        X = torch.relu(self._conv1(X))
        X = torch.relu(self._conv2(X))
        X = torch.relu(self._conv3(X))
        X = torch.relu(self._conv4(X))
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

        X_flat = X_conv.reshape(N * T, -1)
        X_proj: Tensor = torch.relu(self._fc(X_flat))  # [N * T, D]
        X_proj = X_proj.view(N, T, self._hidden_dim)

        gru_out: Tensor  # [N, T, D]
        gru_out, h = self._gru(
            X_proj,
            # nn.GRU expects (L, N, D).
            prev_h.transpose(0, 1).contiguous(),
        )

        gru_out = gru_out.reshape(N * T, -1)
        logits_nonsequential: Tensor = self._policy_head(  # [N * T, V]
            gru_out
        )
        logits = logits_nonsequential.view(N, T, self._vocab_size)

        return logits, h.transpose(0, 1).contiguous()


# from type_defs import ConvBias, ConvWeight
#
# class PolicyModel(nn.Module):
#     """Lightweight 3 layer CNN with a GRU policy. Excludes padding and dilation."""
#
#     def __init__(self):
#         """Initialize hidden dim, all weights and biases, and calculate flat size."""
#         super().__init__()
#         self._hidden_dim = _CONFIG_MODEL["hiddenDim"]
#         self._vocab_size = _CONFIG_MODEL["vocabSize"]
#
#         self._conv1_shape = _CONFIG_MODEL["weightShapes"]["conv1"]
#         self._conv1_W: ConvWeight = nn.Parameter(
#             torch.empty(
#                 self._conv1_shape["kernelSize"],
#                 self._conv1_shape["kernelSize"],
#                 self._conv1_shape["inChannels"],
#                 self._conv1_shape["outChannels"],
#             ),
#         )
#         # Initialize the weights with values following a distribution for stability.
#         nn.init.normal_(self._conv1_W, mean=0.0, std=0.02)
#         self._conv1_b: ConvBias = nn.Parameter(
#             torch.zeros(self._conv1_shape["outChannels"]),
#         )
#
#         self._conv2_shape = _CONFIG_MODEL["weightShapes"]["conv2"]
#         self._conv2_W: ConvWeight = nn.Parameter(
#             torch.empty(
#                 self._conv2_shape["kernelSize"],
#                 self._conv2_shape["kernelSize"],
#                 self._conv2_shape["inChannels"],
#                 self._conv2_shape["outChannels"],
#             ),
#         )
#         nn.init.normal_(self._conv2_W, mean=0.0, std=0.02)
#         self._conv2_b: ConvBias = nn.Parameter(
#             torch.zeros(self._conv2_shape["outChannels"]),
#         )
#
#         self._conv3_shape = _CONFIG_MODEL["weightShapes"]["conv3"]
#         self._conv3_W: ConvWeight = nn.Parameter(
#             torch.empty(
#                 self._conv3_shape["kernelSize"],
#                 self._conv3_shape["kernelSize"],
#                 self._conv3_shape["inChannels"],
#                 self._conv3_shape["outChannels"],
#             ),
#         )
#         nn.init.normal_(self._conv3_W, mean=0.0, std=0.02)
#         self._conv3_b: ConvBias = nn.Parameter(
#             torch.zeros(self._conv3_shape["outChannels"]),
#         )
#
#         # Dynamically calculate flattened size.
#         with torch.inference_mode():
#             frame_H_px = _CONFIG["capture"]["frameDims"]["pipelineHeightPx"]
#             frame_W_px = _CONFIG["capture"]["frameDims"]["pipelineWidthPx"]
#
#             dummy = torch.zeros(
#                 1,
#                 frame_H_px,
#                 frame_W_px,
#                 self._conv1_shape["inChannels"],
#             )
#             flat_size = self._conv_forward(dummy).numel()
#
#         self._fc_W: Float32[Tensor, "hidden_dim flat_size"] = nn.Parameter(
#             torch.empty(self._hidden_dim, flat_size),
#         )
#         nn.init.normal_(self._fc_W, mean=0.0, std=0.02)
#         self._fc_b: Float32[Tensor, "hidden_dim"] = nn.Parameter(
#             torch.zeros(self._hidden_dim),
#         )
#
#         self._gru_in_W: Float32[Tensor, "combined_hidden_dim hidden_dim"] = (
#             nn.Parameter(
#                 torch.empty(3 * self._hidden_dim, self._hidden_dim),
#             )
#         )
#         nn.init.normal_(self._gru_in_W, mean=0.0, std=0.02)
#         self._gru_in_b: Float32[Tensor, "combined_hidden_dim"] = nn.Parameter(
#             torch.zeros(3 * self._hidden_dim),
#         )
#
#         self._gru_hidden_W: Float32[Tensor, "combined_hidden_dim hidden_dim"] = (
#             nn.Parameter(
#                 torch.empty(3 * self._hidden_dim, self._hidden_dim),
#             )
#         )
#         nn.init.normal_(self._gru_hidden_W, mean=0.0, std=0.02)
#         self._gru_hidden_b: Float32[Tensor, "combined_hidden_dim"] = nn.Parameter(
#             torch.zeros(3 * self._hidden_dim),
#         )
#
#         self._policy_W: Float32[Tensor, "out_heads hidden_dim"] = nn.Parameter(
#             torch.empty(self._vocab_size, self._hidden_dim),
#         )
#         nn.init.normal_(self._policy_W, mean=0.0, std=0.02)
#         self._policy_b: Float32[Tensor, "out_heads"] = nn.Parameter(
#             torch.zeros(self._vocab_size),
#         )
#
#
#     @staticmethod
#     def _calculate_out_dim(
#         H: int,
#         W: int,
#         kernel_size: int,
#         stride: int,
#     ) -> tuple[int, int]:
#         """Calculate the feature map dimensions.
#
#         Returns:
#             Output height, output width.
#         """
#         return ((H - kernel_size) // stride + 1, (W - kernel_size) // stride + 1)
#
#     def _extract_patches(
#         self,
#         X: Float32[Tensor, "N H W C"],
#         kernel_size: int,
#         stride: int,
#     ) -> tuple[Tensor, int, int]:
#         """Use for square kernel sliding and reshaping.
#
#         Returns:
#             The strided x with shape (N, H_out, W_out, kernel_size, kernel_size, C), output height and width.
#         """
#         N, H, W, C = X.shape
#
#         H_out, W_out = self._calculate_out_dim(
#             H=H,
#             W=W,
#             kernel_size=kernel_size,
#             stride=stride,
#         )
#
#         # Base strides used to move by each element in memory.
#         N_stride, H_stride, W_stride, C_stride = X.stride()
#
#         X_strided = X.as_strided(
#             size=(N, H_out, W_out, kernel_size, kernel_size, C),
#             stride=(
#                 N_stride,
#                 H_stride * stride,
#                 W_stride * stride,
#                 H_stride,
#                 W_stride,
#                 C_stride,
#             ),
#         )
#
#         return X_strided, H_out, W_out
#
#     def _maxpool(
#         self,
#         X: Float32[Tensor, "N H W C"],
#         kernel_size: int,
#         stride: int,
#     ) -> Float32[Tensor, "N H_out W_out C"]:
#         """Get the maximum pixel values per receptive field.
#
#         Returns:
#             the output feature map where the coordinate value is the maximum of the patch.
#         """
#         patches, _, _ = self._extract_patches(X, kernel_size=kernel_size, stride=stride)
#
#         return patches.amax(dim=(3, 4))
#
#     def _conv(
#         self,
#         X: Float32[Tensor, "N H W C"],
#         W: Float32[Tensor, "kernel_size kernel_size C_in C_out"],
#         b: Float32[Tensor, "C_out"],
#         stride: int,
#     ) -> Float32[Tensor, "N H_out W_out C_out"]:
#         """Multiply weights with unfolded image tensor and adds bias."""
#         N, _, _, C = X.shape
#         kernel_size, _, _, C_out = W.shape
#
#         patches, H_out, width_out = self._extract_patches(
#             X,
#             kernel_size=kernel_size,
#             stride=stride,
#         )
#
#         # Shaped this way so the weight matrix dots the entire receptive field per output feature instead of dots every relative pixel across all kernels.
#         patch_col = patches.reshape(
#             N * H_out * width_out,
#             C * kernel_size * kernel_size,
#         )
#
#         W_col = W.view(-1, C_out)
#
#         # Every value in each channel gets offset by the same amount.
#         out: Float32[Tensor, "all_px C_out"] = patch_col @ W_col + b
#
#         return out.view(N, H_out, width_out, C_out)
#
#     def _conv_forward(
#         self,
#         X: Float32[Tensor, "N H W C"],
#     ) -> Float32[Tensor, "N H W C"]:
#         """3 convolution layers, 2 max pooling layers, ReLU activation."""
#         X = torch.relu(
#             self._conv(
#                 X,
#                 W=self._conv1_W,
#                 b=self._conv1_b,
#                 stride=self._conv1_shape["stride"],
#             ),
#         )
#         X = self._maxpool(X, kernel_size=2, stride=2)
#
#         X = torch.relu(
#             self._conv(
#                 X,
#                 W=self._conv2_W,
#                 b=self._conv2_b,
#                 stride=self._conv2_shape["stride"],
#             ),
#         )
#         X = self._maxpool(X, kernel_size=2, stride=2)
#
#         return torch.relu(
#             self._conv(
#                 X,
#                 W=self._conv3_W,
#                 b=self._conv3_b,
#                 stride=self._conv3_shape["stride"],
#             ),
#         )
#
#     def _gru_step(
#         self,
#         X: Float32[Tensor, "N D"],
#         prev_h: Float32[Tensor, "N D"],
#     ) -> Float32[Tensor, "N D"]:
#         """Matmul with concatenated weights, then split into gates for activation.
#
#         Returns:
#             The new hidden state.
#         """
#         in_combined: Float32[Tensor, "N combined_D"] = (
#             X @ self._gru_in_W.T + self._gru_in_b
#         )
#         hidden_combined: Float32[Tensor, "N combined_D"] = (
#             prev_h @ self._gru_hidden_W.T + self._gru_hidden_b
#         )
#
#         # Split the combined input and hidden tensors into 3 chunks along the combined hidden dim (3 * D) axes.
#         r_in, z_in, h_tilde_in = in_combined.chunk(3, dim=-1)
#         r_hidden, z_hidden, h_tilde_hidden = hidden_combined.chunk(3, dim=-1)
#
#         r = torch.sigmoid(r_in + r_hidden)
#         z = torch.sigmoid(z_in + z_hidden)
#         h_tilde = torch.tanh(h_tilde_in + h_tilde_hidden * r)
#
#         return (1 - z) * h_tilde + prev_h * z
#
#     def forward(
#         self,
#         X: Float32[Tensor, "N T H W C"],
#         prev_hidden_state: Float32[Tensor, "N L D"],
#     ) -> tuple[Float32[Tensor, "N T V"], Float32[Tensor, "N L D"]]:
#         """Pass inputs through CNN + GRU.
#
#         Returns:
#             Logits and the new hidden state.
#         """
#         N, T, H, W, C = X.shape
#
#         # Convolve frame with combined batch size and time since convolution isn't sequential.
#         X = X.view(N * T, H, W, C)
#         X_conv = self._conv_forward(X)
#
#         X_flat = X_conv.view(N * T, -1)
#
#         X_proj: Float32[Tensor, "N_nonsequential D"] = torch.relu(
#             X_flat @ self._fc_W.T + self._fc_b,
#         )
#
#         X_proj = X_proj.view(N, T, self._hidden_dim)
#
#         # Extract hidden_state from layer.
#         h = prev_hidden_state[:, 0, :]
#
#         h_states = []
#
#         for t in range(T):
#             # Only pass the input tensor for the current timestep.
#             h = self._gru_step(X_proj[:, t, :], prev_h=h)
#
#             # Add a time axis for sequential processing.
#             h_states.append(h.unsqueeze(1))
#
#         gru_out: Float32[Tensor, "N T D"] = torch.cat(h_states, dim=1)
#
#         gru_out = gru_out.view(N * T, self._hidden_dim)
#
#         logits_nonsequential: Float32[Tensor, "N_nonsequential V"] = (
#             gru_out @ self._policy_W.T + self._policy_b
#         )
#
#         logits = logits_nonsequential.view(N, T, self._vocab_size)
#
#         # Hidden state needs the L axis back.
#         return logits, h.unsqueeze(1)
