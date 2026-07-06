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
DEVICE = torch.device("mps")


class _DatasetGenerator(IterableDataset):
    """Yield training batches built from parallel gameplay streams."""

    def __init__(self, src_files: list[Path], is_val: bool):
        self.src_files: list[Path] = src_files
        self.is_val: bool = is_val
        self._dataset_files: list[Path] = []

    def _get_next_file(self) -> Path:
        if not self._dataset_files:
            self._dataset_files = self.src_files.copy()
            if not self.is_val:
                random.shuffle(self._dataset_files)
        return self._dataset_files.pop(0)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch together mini batches from each file stream.

        Yields:
            Tuple of arrays of frames, action binaries, and is_first flags of the concatenated mini batches.
        """
        self._dataset_files = []

        file_streams = [self._stream_file(self._get_next_file()) for _ in range(BATCH_SIZE)]

        while True:
            batch_frames = []
            batch_actions_bin = []
            batch_are_first = []

            for batch_idx in range(BATCH_SIZE):
                try:
                    frames, actions_bin, is_first = next(file_streams[batch_idx])
                except StopIteration:
                    file_streams[batch_idx] = self._stream_file(self._get_next_file())
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
            frames: np.ndarray = data["frames"]
            actions_bin: np.ndarray = data["actions_bin"]

            num_chunks = len(frames) // _CONFIG_TRAINING["seqLen"]

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

            # Chop up each file stream into chunks with length seq_len.
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * _CONFIG_TRAINING["seqLen"]
                chunk_frames = frames[start_idx : start_idx + _CONFIG_TRAINING["seqLen"]]
                chunk_actions_bin = actions_bin[start_idx : start_idx + _CONFIG_TRAINING["seqLen"]]

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


def _preprocess_inputs(
    frames: Tensor,
    actions_bin: Tensor,
    are_first: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Preprocess and transfer frames, action targets, and hidden states to the target device.

    Args:
        frames: [N, T, H, W, C] raw frames.
        actions_bin: [N, T, 4] action target binaries.
        are_first: [N] is_first flags.
        hidden: [N, L, D] previous hidden state.

    Returns:
        Normalized frames, action targets, and masked hidden state.
    """
    keep_hidden_mask = (~are_first).to(DEVICE, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    hidden_state = hidden * keep_hidden_mask

    frames_norm = frames.to(DEVICE, non_blocking=True).to(dtype=torch.float32).mul_(1.0 / 255.0)
    target_actions_bin = actions_bin.to(DEVICE, dtype=torch.long)

    return frames_norm, target_actions_bin, hidden_state


def _calculate_loss(
    logits: Tensor,
    target_actions_bin: Tensor,
    class_weights: Tensor,
) -> Tensor:
    """Calculate the loss given model logits and target actions.

    Args:
        logits: [N, T, V] model output logits.
        target_actions_bin: [N, T, 4] target actions on the device.
        class_weights: Class weights tensor.

    Returns:
        The calculated loss tensor.
    """
    N, T, _ = logits.shape
    logits_240 = logits.view(N, T * 4, 2)
    target_actions_bin_240 = target_actions_bin.view(N, T * 4)

    log_probs = F.log_softmax(logits_240, dim=-1)
    log_p_no_jump = log_probs[..., 0]  # [N, T * 4]
    log_p_jump = log_probs[..., 1].unsqueeze(1)  # [N, 1, T * 4]

    kernel_size = _CONFIG_TRAINING["distributionSize"]
    padding = kernel_size // 2
    max_log_p_jump = F.max_pool1d(log_p_jump, kernel_size=kernel_size, stride=1, padding=padding).squeeze(1)

    is_jump = target_actions_bin_240.to(dtype=torch.float32).unsqueeze(1)

    # Use > 0.5 rather than == 1 to mitigate floating point precision issues.
    in_window = F.max_pool1d(is_jump, kernel_size=kernel_size, stride=1, padding=padding).squeeze(1) > 0.5

    loss_no_jump = -log_p_no_jump * (~in_window)
    loss_jump = -max_log_p_jump * (target_actions_bin_240 == 1)

    loss = (torch.sum(loss_no_jump * class_weights[0]) + torch.sum(loss_jump * class_weights[1])) / (
        (~in_window).sum() * class_weights[0] + (target_actions_bin_240 == 1).sum() * class_weights[1] + 1e-8
    )

    return loss


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
        actions_bin: [N, T, 4].
        are_first: [N].
        hidden: [N, L, D].

    Returns:
        A tuple of the loss and new hidden state.
    """
    class_weights = torch.tensor(_CONFIG_TRAINING["classWeights"], device=DEVICE, dtype=torch.float32)

    frames_norm, target_actions_bin, hidden_state = _preprocess_inputs(
        frames=frames,
        actions_bin=actions_bin,
        are_first=are_first,
        hidden=hidden,
    )

    logits, hidden_state = model(frames_norm, hidden_state)

    # Ensure hidden state doesn't effect the gradients of the entire dataset.
    hidden_state = hidden_state.detach()

    loss = _calculate_loss(
        logits=logits,
        target_actions_bin=target_actions_bin,
        class_weights=class_weights,
    )

    return loss, hidden_state


def _prepare_data_files() -> tuple[list[Path], list[Path], int, int, int]:
    """Prepare train/validation files and calculate step limits.

    Returns:
        A tuple containing (train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch).
    """
    dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
    training_dir_name = _CONFIG["fileNames"]["trainingDirName"]
    validation_dir_name = _CONFIG["fileNames"]["validationDirName"]

    dataset_files_src = Path(__file__).resolve().parents[2] / dataset_dir_name
    train_files = list((dataset_files_src / training_dir_name).glob("*.npz"))
    val_files = list((dataset_files_src / validation_dir_name).glob("*.npz"))

    total_train_chunks = sum(len(np.load(f)["actions_bin"]) // _CONFIG_TRAINING["seqLen"] for f in train_files)
    train_steps_per_epoch = max(1, total_train_chunks // BATCH_SIZE)

    total_val_chunks = sum(len(np.load(f)["actions_bin"]) // _CONFIG_TRAINING["seqLen"] for f in val_files)
    val_steps_per_epoch = max(1, total_val_chunks // BATCH_SIZE)

    accumulation_steps: int = _CONFIG["training"]["accumulationSteps"]
    opt_steps_per_epoch = (train_steps_per_epoch + accumulation_steps - 1) // accumulation_steps

    return train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch


def _init_model_and_optimizer(
    opt_steps_per_epoch: int,
) -> tuple[Model, torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    """Initialize Model, Optimizer, and LR Scheduler on the specified device."""
    model = Model().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(_CONFIG_TRAINING["beta1"], _CONFIG_TRAINING["beta2"]),
        weight_decay=_CONFIG_TRAINING["weightDecay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=_CONFIG["training"]["epochs"],
        steps_per_epoch=opt_steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    return model, optimizer, scheduler


def _load_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
) -> int:
    """Load checkpoint state if a checkpoint name is specified.

    Returns:
        The starting epoch number (defaults to 1 if no checkpoint is loaded).
    """
    if not checkpoint_name:
        return 1

    checkpoint = torch.load(
        checkpoint_dir / checkpoint_name,
        map_location=DEVICE,
    )

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"Loading checkpoint {checkpoint_name}.")
    return checkpoint["epoch"] + 1


def _run_train_epoch(
    model: Model,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    steps: int,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    num_train_batches = 0
    train_loss_tensor = torch.zeros(1, device=DEVICE)

    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    hidden_state = torch.zeros(  # [N, L, D]
        BATCH_SIZE,
        1,
        hidden_state_dim,
        device=DEVICE,
    )

    accumulation_steps = _CONFIG["training"]["accumulationSteps"]

    for i in range(steps):
        try:
            frames, actions_bin, are_first = next(train_iter)
        except StopIteration:
            break
        num_train_batches = i + 1

        loss, hidden_state = _process_batch(
            model=model,
            frames=frames,
            actions_bin=actions_bin,
            are_first=are_first,
            hidden=hidden_state,
        )

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

    return (train_loss_tensor.item() / num_train_batches) if num_train_batches > 0 else 0.0


def _run_val_epoch(
    model: Model,
    val_iter: Iterator,
    steps: int,
) -> float:
    """Run a single validation epoch and return the average loss."""
    model.eval()
    num_val_batches = 0
    val_loss_tensor = torch.zeros(1, device=DEVICE)

    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    hidden_state = torch.zeros(  # [N, L, D]
        BATCH_SIZE,
        1,
        hidden_state_dim,
        device=DEVICE,
    )

    with torch.no_grad():
        for i in range(steps):
            try:
                frames, actions_bin, are_first = next(val_iter)
            except StopIteration:
                break
            num_val_batches = i + 1

            loss, hidden_state = _process_batch(
                model=model,
                frames=frames,
                actions_bin=actions_bin,
                are_first=are_first,
                hidden=hidden_state,
            )

            val_loss_tensor += loss.detach()

    return (val_loss_tensor.item() / num_val_batches) if num_val_batches > 0 else 0.0


def _save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    train_loss: float,
    val_loss: float,
) -> None:
    """Save the model checkpoint."""
    checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        },
        checkpoint_path,
    )
    print(f"Checkpoint saved to {checkpoint_path}")


def _train():
    """Load model, previous checkpoints, and dataset. Train over epochs hyper-parameter."""
    checkpoint_save_interval: int = _CONFIG["training"]["checkpointSaveInterval"]

    train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch = _prepare_data_files()

    dataloader = DataLoader(_DatasetGenerator(train_files, is_val=False), batch_size=None)
    dataloader_validation = DataLoader(_DatasetGenerator(val_files, is_val=True), batch_size=None)

    train_iter = iter(dataloader)
    val_iter = iter(dataloader_validation)

    model, optimizer, scheduler = _init_model_and_optimizer(
        opt_steps_per_epoch=opt_steps_per_epoch,
    )

    checkpoint_dir = Path(__file__).resolve().parents[2] / _CONFIG["fileNames"]["checkpointDirName"]
    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]

    start_epoch = _load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    for epoch in range(start_epoch, _CONFIG["training"]["epochs"] + 1):
        avg_train_loss = _run_train_epoch(
            model=model,
            train_iter=train_iter,
            optimizer=optimizer,
            scheduler=scheduler,
            steps=train_steps_per_epoch,
        )

        avg_val_loss = _run_val_epoch(
            model=model,
            val_iter=val_iter,
            steps=val_steps_per_epoch,
        )

        print(f"Epoch {epoch} completed | Training loss: {avg_train_loss:.2f} | Validation loss: {avg_val_loss:.2f}")

        if epoch % checkpoint_save_interval == 0:
            _save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss=avg_train_loss,
                val_loss=avg_val_loss,
            )


if __name__ == "__main__":
    _train()
