"""Handles logic from loading the saved data files, to loading the previous checkpoint and continuing training.

Example:
    $ python train.py
"""

import json
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.model import Model

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
    _CONFIG = json.load(f)

_CONFIG_TRAINING = _CONFIG["training"]

BATCH_SIZE: int = _CONFIG_TRAINING["batchSize"]
LR = _CONFIG_TRAINING["learningRate"]


class _DatasetGenerator(IterableDataset):
    """Yield training batches built from parallel gameplay streams."""

    def __init__(self, files: list[Path]):
        self.files = files

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch together mini batches from each file stream.

        Yields:
            Tuple of arrays of frames, action binaries, and is_first flags of the concatenated mini batches.
        """

        dataset_files: list[Path] = self.files.copy()
        random.shuffle(self.files)

        file_streams: list[Iterator[tuple[np.ndarray, np.ndarray, bool]]] = [
            self._stream_file(dataset_files.pop(0)) for _ in range(BATCH_SIZE)
        ]

        while True:
            active_stream_count = 0

            batch_frames: list[np.ndarray] = []
            batch_actions_bin: list[np.ndarray] = []
            batch_are_first: list[bool] = []

            for batch_idx, curr_file_stream in enumerate(file_streams):
                frames: np.ndarray
                actions_bin: np.ndarray
                is_first: bool

                try:
                    frames, actions_bin, is_first = next(curr_file_stream)
                except StopIteration:
                    # If current file is exhausted, attempt to refill slot with a new file.
                    if dataset_files:
                        file_streams[batch_idx] = self._stream_file(
                            dataset_files.pop(0),
                        )

                        frames, actions_bin, is_first = next(
                            file_streams[batch_idx],
                        )
                    else:
                        break

                batch_frames.append(frames)
                batch_actions_bin.append(actions_bin)
                batch_are_first.append(is_first)

                active_stream_count += 1

            if active_stream_count < BATCH_SIZE:
                break

            yield (
                np.stack(batch_frames),
                np.stack(batch_actions_bin),
                np.stack(batch_are_first),
            )

    def _stream_file(
        self,
        filepath: Path,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, bool]]:
        """Extract all frames and actions from the entire file, then chop them up into chunks to pass back.

        Yields:
            The frames, action binaries, and whether the chunk is the first of the file.
        """
        with np.load(filepath) as data:
            frames: np.ndarray
            actions_bin: np.ndarray
            frames, actions_bin = data["frames"], data["actions_bin"]

            num_chunks = (len(frames) - 1) // _CONFIG_TRAINING["seqLen"]

            # Chop up each file stream into chunks with length seq_len.
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * _CONFIG_TRAINING["seqLen"]

                chunk_frames: np.ndarray = frames[start_idx : start_idx + _CONFIG_TRAINING["seqLen"]]
                chunk_actions_bin: np.ndarray = actions_bin[start_idx : start_idx + _CONFIG_TRAINING["seqLen"]]

                yield chunk_frames, chunk_actions_bin, (chunk_idx == 0)


# class _AdamW:
#     """Manual implementation of the AdamW optimizer."""
#
#     def __init__(self, params):
#         self._params = list(params)
#         self._m = [torch.zeros_like(p) for p in self._params]
#         self._v = [torch.zeros_like(p) for p in self._params]
#         self.step_idx = 1
#
#     @torch.no_grad()
#     def step(self):
#         beta1 = _CONFIG_TRAINING["beta1"]
#         beta2 = _CONFIG_TRAINING["beta2"]
#         W_decay = _CONFIG_TRAINING["weightDecay"]
#
#         for i, param in enumerate(self._params):
#             grad = param.grad
#             if grad is None:
#                 raise Exception("Parameter has no gradient.")
#
#             self._m[i] = beta1 * self._m[i] + (1 - beta1) * grad
#             self._v[i] = beta2 * self._v[i] + (1 - beta2) * (grad**2)
#
#             m_hat = self._m[i] / (1 - beta1**self.step_idx)
#             v_hat = self._v[i] / (1 - beta2**self.step_idx)
#
#             param_adam = self._params[i] - LR * m_hat / (torch.sqrt(v_hat) + 1e-8)
#             self._params[i].copy_(param_adam - LR * W_decay * param)
#
#         self.step_idx += 1
#
#     def clear_grad(self):
#         for param in self._params:
#             param.grad = None
#
#     def get_state_dict(self):
#         return {"step_idx": self.step_idx, "mean": self._m, "var": self._v}
#
#     def load_state_dict(self, state_dict):
#         self.step_idx = state_dict["step_idx"]
#         self._m = state_dict["mean"]
#         self._v = state_dict["var"]


def _process_batch(
    model: Model,
    frames: Tensor,
    actions_bin: Tensor,
    are_first: Tensor,
    hidden: Tensor,
    class_weights: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Process a single batch through the model and calculate loss.

    Args:
        model: the neural network.
        frames: [N, T, H, W].
        actions_bin: [N, T].
        are_first: [N].
        hidden: [N, L, D].
        class_weights: [V].
        device: The device.

    Returns:
        A tuple of the loss and new hidden state.
    """
    vocab_size: int = _CONFIG["model"]["vocabSize"]

    keep_hidden_mask = ~are_first
    keep_hidden_mask = (
        keep_hidden_mask.to(
            device,
            dtype=torch.float32,
        )
        .unsqueeze(-1)
        .unsqueeze(-1)
    )

    hidden_state = hidden * keep_hidden_mask

    frames = frames.to(device, non_blocking=True)
    frames_norm = frames.to(dtype=torch.float32).mul_(1.0 / 255.0)
    target_actions_bin = actions_bin.to(
        device,
        dtype=torch.long,
    )

    logits: Tensor  # [N, T, V]
    hidden_state: Tensor  # [N, L, D]
    logits, hidden_state = model(frames_norm, hidden_state)

    # Ensure hidden state doesn't effect the gradients of the entire dataset.
    hidden_state = hidden_state.detach()

    target_one_hot = F.one_hot(target_actions_bin, num_classes=vocab_size).to(dtype=torch.float32)
    kernel = (
        torch.tensor([0.02, 0.08, 0.2, 0.4, 0.2, 0.08, 0.02], device=device, dtype=torch.float32)
        .view(1, 1, 7)
        .repeat(vocab_size, 1, 1)
    )

    smoothed_targets = F.conv1d(target_one_hot.transpose(1, 2), kernel, padding=3, groups=vocab_size)
    smoothed_targets = smoothed_targets.transpose(1, 2)

    # Normalize because of edge padding.
    smoothed_targets = smoothed_targets / smoothed_targets.sum(dim=-1, keepdim=True)

    logits: Tensor = logits.reshape(-1, vocab_size)
    target_actions: Tensor = smoothed_targets.reshape(-1, vocab_size)

    loss: Tensor = F.cross_entropy(
        logits,
        target_actions,
        weight=class_weights,
    )

    return loss, hidden_state


def _train():
    """Load model, previous checkpoints, and dataset. Train over epochs hyper-parameter."""
    device: torch.device = torch.device(_CONFIG["model"]["deviceName"])
    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    epochs = _CONFIG["training"]["epochs"]
    checkpoint_save_interval: int = _CONFIG["training"]["checkpointSaveInterval"]

    dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
    dataset_files_src: Path = Path(__file__).resolve().parents[2] / dataset_dir_name
    dataset: list[Path] = list(dataset_files_src.glob("*.npz"))[2:]
    validation_set: list[Path] = list(dataset_files_src.glob("*.npz"))[:2]

    dataloader: DataLoader = DataLoader(_DatasetGenerator(dataset), batch_size=None)
    dataloader_validation: DataLoader = DataLoader(_DatasetGenerator(validation_set), batch_size=None)
    model: Model = Model().to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(_CONFIG_TRAINING["beta1"], _CONFIG_TRAINING["beta2"]),
        weight_decay=_CONFIG_TRAINING["weightDecay"],
    )

    checkpoint_dir = Path(__file__).resolve().parents[2] / _CONFIG["fileNames"]["checkpointDirName"]
    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]

    if checkpoint_name:
        checkpoint: dict[str, int | float | dict[str, int | Tensor]] = torch.load(
            checkpoint_dir / checkpoint_name,
            map_location=device,
        )

        start_epoch: int = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        print(f"Loading checkpoint {checkpoint_name}.")
    else:
        start_epoch: int = 1

    for epoch in range(start_epoch, epochs + 1):
        class_weights = torch.tensor(_CONFIG_TRAINING["classWeights"], device=device, dtype=torch.float32)

        model.train()
        num_train_batches = 0
        train_loss_tensor: Tensor = torch.zeros(1, device=device)
        hidden_state: torch.Tensor = torch.zeros(  # [N, L, D]
            BATCH_SIZE,
            1,
            hidden_state_dim,
            device=device,
        )

        for i, (frames, actions_bin, are_first) in enumerate(dataloader):
            num_train_batches = i + 1

            loss, hidden_state = _process_batch(
                model, frames, actions_bin, are_first, hidden_state, class_weights, device
            )

            accumulation_steps = _CONFIG["training"]["accumulationSteps"]

            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_tensor += loss.detach()

        avg_train_loss = (train_loss_tensor.item() / num_train_batches) if num_train_batches > 0 else 0

        model.eval()
        num_val_batches = 0
        val_loss_tensor: Tensor = torch.zeros(1, device=device)
        hidden_state: torch.Tensor = torch.zeros(  # [N, L, D]
            BATCH_SIZE,
            1,
            hidden_state_dim,
            device=device,
        )

        with torch.no_grad():
            for i, (frames, actions_bin, are_first) in enumerate(dataloader_validation):
                num_val_batches = i + 1

                loss, hidden_state = _process_batch(
                    model, frames, actions_bin, are_first, hidden_state, class_weights, device
                )

                val_loss_tensor += loss.detach()

        avg_val_loss = (val_loss_tensor.item() / num_val_batches) if num_val_batches > 0 else 0

        print(f"Epoch {epoch} completed | Train loss: {avg_train_loss:.2f} | Validation loss: {avg_val_loss:.2f}")

        if epoch % checkpoint_save_interval == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    _train()
