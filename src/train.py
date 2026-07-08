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

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model

with (Path(__file__).resolve().parent / "config.json").open() as f:
    _CONFIG = json.load(f)

_BATCH_SIZE = 4
_SEQ_LEN = 128
_ACCUMULATION_STEPS = 4
_EPOCHS = 100

_DEVICE = torch.device("mps")


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

        file_streams = [self._stream_file(self._get_next_file()) for _ in range(_BATCH_SIZE)]

        while True:
            batch_frames = []
            batch_actions_bin = []
            batch_are_first = []

            for batch_idx in range(_BATCH_SIZE):
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

            num_chunks = len(frames) // _SEQ_LEN

            # Chop up each file stream into chunks with length seq_len.
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * _SEQ_LEN
                chunk_frames = frames[start_idx : start_idx + _SEQ_LEN]
                chunk_actions_bin = actions_bin[start_idx : start_idx + _SEQ_LEN]

                yield chunk_frames, chunk_actions_bin, (chunk_idx == 0)


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
    keep_hidden_mask = (~are_first).to(_DEVICE, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
    hidden_state = hidden * keep_hidden_mask

    frames_norm = frames.to(_DEVICE, non_blocking=True).to(dtype=torch.float32).mul_(1.0 / 255.0)
    target_actions_bin = actions_bin.to(_DEVICE, dtype=torch.long)

    return frames_norm, target_actions_bin, hidden_state


def _calculate_loss(
    logits: Tensor,
    target_actions_bin: Tensor,
) -> Tensor:
    """Calculate the loss given model logits and target actions.

    Args:
        logits: [N, T, V] model output logits.
        target_actions_bin: [N, T, 4] target actions on the device.

    Returns:
        The calculated loss tensor.
    """
    N, T, _ = logits.shape
    logits_240 = logits.view(N, T * 4, 2)
    target_actions_bin_240 = target_actions_bin.view(N, T * 4)

    num_no_jump = (target_actions_bin_240 == 0).sum()
    num_jump = (target_actions_bin_240 == 1).sum()

    # Avoid division by zero if a class is entirely missing in the batch
    w_no_jump = 1.0 / (num_no_jump.clamp(min=1).to(dtype=torch.float32))
    w_jump = 1.0 / (num_jump.clamp(min=1).to(dtype=torch.float32))

    # If a class is missing, set its weight to 0.0
    weight = torch.stack([w_no_jump * (num_no_jump > 0), w_jump * (num_jump > 0)])

    loss = F.cross_entropy(logits_240.transpose(1, 2), target_actions_bin_240, weight=weight)

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
    )

    return loss, hidden_state


def _prepare_data_files() -> tuple[list[Path], list[Path], int, int, int]:
    """Prepare train/validation files and calculate step limits.

    Returns:
        A tuple containing (train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch).
    """
    repo_dir = Path(__file__).resolve().parents[1]
    train_files = list((repo_dir / "data" / "training").glob("*.npz"))
    val_files = list((repo_dir / "data" / "validation").glob("*.npz"))

    def get_chunks(f: Path) -> int:
        with np.load(f) as data:
            return len(data["actions_bin"]) // _SEQ_LEN

    total_train_chunks = sum(get_chunks(f) for f in train_files)
    train_steps_per_epoch = max(1, total_train_chunks // _BATCH_SIZE)

    total_val_chunks = sum(get_chunks(f) for f in val_files)
    val_steps_per_epoch = max(1, total_val_chunks // _BATCH_SIZE)

    opt_steps_per_epoch = (train_steps_per_epoch + _ACCUMULATION_STEPS - 1) // _ACCUMULATION_STEPS

    return train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch


def _init_model_and_optimizer(
    opt_steps_per_epoch: int,
) -> tuple[Model, torch.optim.Optimizer, torch.optim.lr_scheduler.OneCycleLR]:
    """Initialize Model, Optimizer, and LR Scheduler on the specified device."""
    model = Model().to(_DEVICE)

    lr = 3e-4
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=_EPOCHS,
        steps_per_epoch=opt_steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy="cos",
    )

    return model, optimizer, scheduler


def _load_checkpoint(
    checkpoint_dir: Path,
    checkpoint_file: str,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
) -> int:
    """Load checkpoint state if a checkpoint file is specified.

    Returns:
        The starting epoch number (defaults to 1 if no checkpoint is loaded).
    """
    if not checkpoint_file:
        return 1

    checkpoint = torch.load(
        checkpoint_dir / checkpoint_file,
        map_location=_DEVICE,
    )

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"Loading checkpoint {checkpoint_file}.")
    return checkpoint["epoch"] + 1


def _run_train_epoch(
    model: Model,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    steps: int,
    hidden_state: Tensor,
) -> tuple[float, Tensor]:
    """Run a single training epoch and return the average loss and final hidden state."""
    model.train()
    num_train_batches = 0
    train_loss_tensor = torch.zeros(1, device=_DEVICE)

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

        scaled_loss = loss / _ACCUMULATION_STEPS
        scaled_loss.backward()

        if num_train_batches % _ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss_tensor += loss.detach()

    if num_train_batches % _ACCUMULATION_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    return ((train_loss_tensor.item() / num_train_batches) if num_train_batches > 0 else 0.0), hidden_state


def _run_val_epoch(
    model: Model,
    val_iter: Iterator,
    steps: int,
    hidden_state: Tensor,
) -> tuple[float, Tensor]:
    """Run a single validation epoch and return the average loss and final hidden state."""
    model.eval()
    num_val_batches = 0
    val_loss_tensor = torch.zeros(1, device=_DEVICE)

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

    return ((val_loss_tensor.item() / num_val_batches) if num_val_batches > 0 else 0.0), hidden_state


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
    epochs = 100

    train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch = _prepare_data_files()

    dataloader = DataLoader(_DatasetGenerator(train_files, is_val=False), batch_size=None)
    dataloader_validation = DataLoader(_DatasetGenerator(val_files, is_val=True), batch_size=None)

    train_iter = iter(dataloader)
    val_iter = iter(dataloader_validation)

    model, optimizer, scheduler = _init_model_and_optimizer(
        opt_steps_per_epoch=opt_steps_per_epoch,
    )

    checkpoint_file = _CONFIG["checkpointFile"]
    checkpoint_dir = Path(__file__).resolve().parents[1] / "checkpoints"

    start_epoch = _load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    hidden_state_dim = _CONFIG["model"]["hiddenDim"]

    # keep the 2 hiddens separate
    train_hidden_state = torch.zeros(
        _BATCH_SIZE,
        1,
        hidden_state_dim,
        device=_DEVICE,
    )
    val_hidden_state = torch.zeros(
        _BATCH_SIZE,
        1,
        hidden_state_dim,
        device=_DEVICE,
    )

    for epoch in range(start_epoch, epochs + 1):
        avg_train_loss, train_hidden_state = _run_train_epoch(
            model=model,
            train_iter=train_iter,
            optimizer=optimizer,
            scheduler=scheduler,
            steps=train_steps_per_epoch,
            hidden_state=train_hidden_state,
        )

        avg_val_loss, val_hidden_state = _run_val_epoch(
            model=model,
            val_iter=val_iter,
            steps=val_steps_per_epoch,
            hidden_state=val_hidden_state,
        )

        print(f"Epoch {epoch} completed | Training loss: {avg_train_loss:.2f} | Validation loss: {avg_val_loss:.2f}")

        if epoch % 10 == 0:
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
