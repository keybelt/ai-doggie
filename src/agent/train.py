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

    def __init__(self, src_files: list[Path], is_val: bool):
        self.src_files: list[Path] = src_files
        self.is_val: bool = is_val

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch together mini batches from each file stream.

        Yields:
            Tuple of arrays of frames, action binaries, and is_first flags of the concatenated mini batches.
        """

        dataset_files: list[Path] = self.src_files.copy()
        random.shuffle(dataset_files)

        def get_next_file() -> Path:
            nonlocal dataset_files
            if not dataset_files:
                dataset_files = self.src_files.copy()
                random.shuffle(dataset_files)
            return dataset_files.pop(0)

        file_streams: list[Iterator[tuple[np.ndarray, np.ndarray, bool]]] = [
            self._stream_file(get_next_file()) for _ in range(BATCH_SIZE)
        ]

        while True:
            batch_frames: list[np.ndarray] = []
            batch_actions_bin: list[np.ndarray] = []
            batch_are_first: list[bool] = []

            for batch_idx in range(BATCH_SIZE):
                frames: np.ndarray
                actions_bin: np.ndarray
                is_first: bool

                try:
                    frames, actions_bin, is_first = next(file_streams[batch_idx])
                except StopIteration:
                    file_streams[batch_idx] = self._stream_file(get_next_file())
                    frames, actions_bin, is_first = next(file_streams[batch_idx])

                batch_frames.append(frames)
                batch_actions_bin.append(actions_bin)
                batch_are_first.append(is_first)

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

            # Per-file translational invariance.
            # _, H, W, _ = frames.shape
            # input_H = _CONFIG["model"]["inputHeightPx"]
            # input_W = _CONFIG["model"]["inputWidthPx"]

            # if self.is_val:
            #     h_offset = (H - input_H) // 2
            #     w_offset = (W - input_W) // 2
            # else:
            #     h_offset = random.randint(0, H - input_H)
            #     w_offset = random.randint(0, W - input_W)
            # frames = frames[:, h_offset : h_offset + input_H, w_offset : w_offset + input_W, :]

            num_chunks = len(frames) // _CONFIG_TRAINING["seqLen"]

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
) -> tuple[Tensor, Tensor]:
    """Process a single batch through the model and calculate loss.

    Args:
        model: the neural network.
        frames: [N, T, H, W, C].
        actions_bin: [N, T].
        are_first: [N].
        hidden: [N, L, D].

    Returns:
        A tuple of the loss and new hidden state.
    """
    device = torch.device("mps")
    class_weights = torch.tensor(_CONFIG_TRAINING["classWeights"], device=device, dtype=torch.float32)

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

    log_probs = F.log_softmax(logits, dim=-1)
    log_p_no_jump = log_probs[..., 0]  # [N, T]
    log_p_jump = log_probs[..., 1].unsqueeze(1)  # [N, 1, T]

    kernel_size = _CONFIG_TRAINING["distributionSize"]
    padding = kernel_size // 2
    max_log_p_jump = F.max_pool1d(log_p_jump, kernel_size=kernel_size, stride=1, padding=padding).squeeze(1)

    is_jump = target_actions_bin.to(dtype=torch.float32).unsqueeze(1)

    # Use > 0.5 rather than == 1 to mitigate floating point precision issues.
    in_window = F.max_pool1d(is_jump, kernel_size=kernel_size, stride=1, padding=padding).squeeze(1) > 0.5

    loss_no_jump = -log_p_no_jump * (~in_window)
    loss_jump = -max_log_p_jump * (target_actions_bin == 1)

    loss = (torch.sum(loss_no_jump * class_weights[0]) + torch.sum(loss_jump * class_weights[1])) / (
        (~in_window).sum() * class_weights[0] + (target_actions_bin == 1).sum() * class_weights[1] + 1e-8
    )

    return loss, hidden_state


def _train():
    """Load model, previous checkpoints, and dataset. Train over epochs hyper-parameter."""
    device: torch.device = torch.device("mps")
    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    epochs = _CONFIG["training"]["epochs"]
    checkpoint_save_interval: int = _CONFIG["training"]["checkpointSaveInterval"]

    dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
    training_dir_name = _CONFIG["fileNames"]["trainingDirName"]
    validation_dir_name = _CONFIG["fileNames"]["validationDirName"]

    dataset_files_src: Path = Path(__file__).resolve().parents[2] / dataset_dir_name
    train_files_src: Path = dataset_files_src / training_dir_name
    val_files_src: Path = dataset_files_src / validation_dir_name

    train_files = list(train_files_src.glob("*.npz"))
    val_files = list(val_files_src.glob("*.npz"))

    total_train_chunks = sum(len(np.load(f)["actions_bin"]) // _CONFIG_TRAINING["seqLen"] for f in train_files)
    train_steps_per_epoch = max(1, total_train_chunks // BATCH_SIZE)

    total_val_chunks = sum(len(np.load(f)["actions_bin"]) // _CONFIG_TRAINING["seqLen"] for f in val_files)
    val_steps_per_epoch = max(1, total_val_chunks // BATCH_SIZE)

    accumulation_steps: int = _CONFIG["training"]["accumulationSteps"]
    opt_steps_per_epoch = (train_steps_per_epoch + accumulation_steps - 1) // accumulation_steps

    dataloader: DataLoader = DataLoader(_DatasetGenerator(train_files, is_val=False), batch_size=None)
    dataloader_validation: DataLoader = DataLoader(_DatasetGenerator(val_files, is_val=True), batch_size=None)

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

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=epochs,
        steps_per_epoch=opt_steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    if checkpoint_name:
        checkpoint: dict[str, int | float | dict[str, int | Tensor]] = torch.load(
            checkpoint_dir / checkpoint_name,
            map_location=device,
        )

        start_epoch: int = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"Loading checkpoint {checkpoint_name}.")
    else:
        start_epoch: int = 1

    for epoch in range(start_epoch, epochs + 1):
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
            if i >= train_steps_per_epoch:
                break
            num_train_batches = i + 1

            loss, hidden_state = _process_batch(model, frames, actions_bin, are_first, hidden_state)

            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()

            if num_train_batches % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_loss_tensor += loss.detach()

        if num_train_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

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
                if i >= val_steps_per_epoch:
                    break
                num_val_batches = i + 1

                loss, hidden_state = _process_batch(model, frames, actions_bin, are_first, hidden_state)

                val_loss_tensor += loss.detach()

        avg_val_loss = (val_loss_tensor.item() / num_val_batches) if num_val_batches > 0 else 0

        print(f"Epoch {epoch} completed | Training loss: {avg_train_loss:.2f} | Validation loss: {avg_val_loss:.2f}")

        if epoch % checkpoint_save_interval == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    _train()
