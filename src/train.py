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
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model

# Shared configuration constants
_BATCH_SIZE = 2
_SEQ_LEN = 128
_ACCUMULATION_STEPS = 16
_EPOCHS = 10
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
    CLASS_WEIGHTS = [1.0, 2.6281]

    frames_norm, target_actions_bin, hidden_state = _preprocess_inputs(
        frames=frames,
        actions_bin=actions_bin,
        are_first=are_first,
        hidden=hidden,
    )

    logits, hidden_state = model(frames_norm, hidden_state)

    # Ensure hidden state doesn't effect the gradients of the entire dataset.
    hidden_state = hidden_state.detach()
    N, T, _ = logits.shape
    logits_240 = logits.view(N, T * 4, 2)
    target_actions_bin_240 = target_actions_bin.view(N, T * 4)

    weight = torch.tensor(CLASS_WEIGHTS, device=logits.device)
    loss = F.cross_entropy(logits_240.transpose(1, 2), target_actions_bin_240, weight=weight)

    return loss, hidden_state


def _prepare_data_files() -> tuple[list[Path], list[Path], int, int, int]:
    """Prepare train/validation files and calculate step limits.

    Returns:
        A tuple containing (train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch).
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TRAINING_DATA_DIR = PROJECT_ROOT / "data" / "training"
    VALIDATION_DATA_DIR = PROJECT_ROOT / "data" / "validation"

    train_files = list(TRAINING_DATA_DIR.glob("*.npz"))
    val_files = list(VALIDATION_DATA_DIR.glob("*.npz"))

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
    LR = 1e-3
    ADAM_BETAS = (0.9, 0.98)
    WEIGHT_DECAY = 0.001
    SCHEDULER_PCT_START = 0.15

    model = Model().to(_DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=ADAM_BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=_EPOCHS,
        steps_per_epoch=opt_steps_per_epoch,
        pct_start=SCHEDULER_PCT_START,
        anneal_strategy="cos",
    )

    return model, optimizer, scheduler


def _load_checkpoint(checkpoint_file: str, state: dict) -> int:
    """Load checkpoint state if a checkpoint file is specified.

    Returns:
        The starting epoch number (defaults to 1 if no checkpoint is loaded).
    """
    if not checkpoint_file:
        return 1

    checkpoint = torch.load(
        Path(checkpoint_file),
        map_location=_DEVICE,
    )

    state["model"].load_state_dict(checkpoint["model_state"])
    state["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    state["scheduler"].load_state_dict(checkpoint["scheduler_state"])

    print(f"Loading checkpoint {checkpoint_file}.")
    return checkpoint["epoch"] + 1


def _log_diagnostics(
    loss: float,
    lr: float,
    val_loss: float,
    global_step: int,
    model: nn.Module,
):
    """Log training metrics, validation loss, and parameter statistics to WandB."""
    stats = {
        "trainer/global_step": global_step,
        "run/train_step_loss": loss,
        "run/learning_rate": lr,
        "run/val_step_loss": val_loss,
    }
    with torch.no_grad():
        for name, param in model.named_parameters():
            stats[f"mean/{name}"] = param.mean().item()
            stats[f"var/{name}"] = param.var().item()
    wandb.log(stats)


def _optimize_and_evaluate(state: dict, loss: Tensor) -> None:
    """Perform optimizer step, learning rate scheduler update, grad norm clipping, and validation."""
    EVAL_FREQ_STEPS = 100
    MAX_GRAD_NORM = 1.0

    model = state["model"]
    optimizer = state["optimizer"]
    scheduler = state["scheduler"]

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    state["global_step"] += 1

    if state["global_step"] % EVAL_FREQ_STEPS == 0:
        val_loss = _run_val(state)
        _log_diagnostics(
            loss=loss.item(),
            lr=scheduler.get_last_lr()[0],
            val_loss=val_loss,
            global_step=state["global_step"],
            model=model,
        )
        model.train()


def _run_train_epoch(state: dict, steps: int) -> float:
    """Run a single training epoch and return training loss."""
    model = state["model"]
    train_iter = state["train_iter"]

    model.train()
    num_train_batches = 0
    train_loss_tensor = torch.zeros(1, device=_DEVICE)
    loss = None

    for i in range(steps):
        try:
            frames, actions_bin, are_first = next(train_iter)
        except StopIteration:
            break
        num_train_batches = i + 1

        loss, state["train_hidden"] = _process_batch(
            model=model,
            frames=frames,
            actions_bin=actions_bin,
            are_first=are_first,
            hidden=state["train_hidden"],
        )

        scaled_loss = loss / _ACCUMULATION_STEPS
        scaled_loss.backward()

        if num_train_batches % _ACCUMULATION_STEPS == 0:
            _optimize_and_evaluate(state, loss)

        train_loss_tensor += loss.detach()

    if num_train_batches % _ACCUMULATION_STEPS != 0 and loss is not None:
        _optimize_and_evaluate(state, loss)

    avg_train_loss = (train_loss_tensor.item() / num_train_batches) if num_train_batches > 0 else 0.0
    return avg_train_loss


def _run_val(state: dict) -> float:
    """Run a fixed number of validation steps and return the average loss."""
    VAL_STEPS = 50

    model = state["model"]
    val_iter = state["val_iter"]

    model.eval()
    num_val_batches = 0
    val_loss_tensor = torch.zeros(1, device=_DEVICE)

    with torch.no_grad():
        for i in range(VAL_STEPS):
            try:
                frames, actions_bin, are_first = next(val_iter)
            except StopIteration:
                break
            num_val_batches = i + 1

            loss, state["val_hidden"] = _process_batch(
                model=model,
                frames=frames,
                actions_bin=actions_bin,
                are_first=are_first,
                hidden=state["val_hidden"],
            )

            val_loss_tensor += loss.detach()

    return (val_loss_tensor.item() / num_val_batches) if num_val_batches > 0 else 0.0


def _save_checkpoint(epoch: int, state: dict, train_loss: float, val_loss: float | None = None) -> None:
    """Save the model checkpoint."""
    checkpoint_data = {
        "epoch": epoch,
        "model_state": state["model"].state_dict(),
        "optimizer_state": state["optimizer"].state_dict(),
        "scheduler_state": state["scheduler"].state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    # Save epoch-specific checkpoint inside WandB run directory for automatic cloud upload
    if wandb.run is not None:
        wandb_checkpoint_path = Path(wandb.run.dir) / f"epoch_{epoch}.pt"
        torch.save(checkpoint_data, wandb_checkpoint_path)


def _train():
    """Load model, previous checkpoints, and dataset. Train over epochs hyper-parameter."""
    WANDB_CONFIG_LR = 3e-4
    WEIGHT_DECAY = 0.001

    train_files, val_files, train_steps_per_epoch, _, opt_steps_per_epoch = _prepare_data_files()

    train_loader = DataLoader(_DatasetGenerator(train_files, is_val=False), batch_size=None)
    val_loader = DataLoader(_DatasetGenerator(val_files, is_val=True), batch_size=None)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    model, optimizer, scheduler = _init_model_and_optimizer(
        opt_steps_per_epoch=opt_steps_per_epoch,
    )

    with (Path(__file__).resolve().parent / "config.json").open() as f:
        config = json.load(f)

    hidden_state_dim = config["model"]["hiddenDim"]

    # Initialize state context to reduce argument passing
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_iter": train_iter,
        "val_iter": val_iter,
        "global_step": 0,
        "train_hidden": torch.zeros(_BATCH_SIZE, 1, hidden_state_dim, device=_DEVICE),
        "val_hidden": torch.zeros(_BATCH_SIZE, 1, hidden_state_dim, device=_DEVICE),
    }

    start_epoch = _load_checkpoint(
        checkpoint_file=config["checkpointFile"],
        state=state,
    )

    state["global_step"] = (start_epoch - 1) * opt_steps_per_epoch

    # Initialize Weights & Biases
    wandb.init(
        project="ai-doggie",
        name="baseline",
        config={
            "batch_size": _BATCH_SIZE,
            "seq_len": _SEQ_LEN,
            "accumulation_steps": _ACCUMULATION_STEPS,
            "epochs": _EPOCHS,
            "learning_rate": WANDB_CONFIG_LR,
            "weight_decay": WEIGHT_DECAY,
            "hidden_dim": hidden_state_dim,
        },
    )

    for epoch in range(start_epoch, _EPOCHS + 1):
        avg_train_loss = _run_train_epoch(
            state=state,
            steps=train_steps_per_epoch,
        )

        if epoch % 10 == 0:
            _save_checkpoint(
                epoch=epoch,
                state=state,
                train_loss=avg_train_loss,
            )

    wandb.finish()


if __name__ == "__main__":
    _train()
