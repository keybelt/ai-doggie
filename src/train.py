import json
import math
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

DEVICE = torch.device("mps")
CLASS_WEIGHTS = torch.tensor(CONFIG["training"]["classWeights"], device=DEVICE)


class DatasetGenerator(IterableDataset):
    """Yield training batches built from parallel gameplay streams."""

    def __init__(self, src_files: list[Path], is_val: bool):
        self.src_files = src_files
        self.is_val = is_val
        self.dataset_files: list[Path] = []

    def get_next_file(self) -> Path:
        if not self.dataset_files:
            self.dataset_files = self.src_files.copy()
            if not self.is_val:
                random.shuffle(self.dataset_files)
        return self.dataset_files.pop(0)

    def stream_file(self, filepath: Path) -> Iterator[tuple[np.ndarray, np.ndarray, bool]]:
        """Stream frame and action chunks from HDF5 file directly without full RAM load.

        Yields:
            Tuple of (frames_chunk, actions_chunk, is_first_chunk_flag).
        """
        seq_len = CONFIG["training"]["seqLen"]
        with h5py.File(filepath, "r") as f:
            frames_ds = f["frames"]
            actions_ds = f["actions_bin"]
            num_chunks = len(frames_ds) // seq_len

            for chunk_idx in range(num_chunks):
                start = chunk_idx * seq_len
                end = start + seq_len
                yield frames_ds[start:end], actions_ds[start:end], (chunk_idx == 0)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch together mini batches from each file stream.

        Yields:
            Tuple of (frames, actions_bin, are_first) concatenated arrays.
        """
        batch_size = CONFIG["training"]["batchSize"]
        self.dataset_files = []
        file_streams = [self.stream_file(self.get_next_file()) for _ in range(batch_size)]

        while True:
            batch_frames, batch_actions_bin, batch_are_first = [], [], []

            for i in range(batch_size):
                try:
                    frames, actions_bin, is_first = next(file_streams[i])
                except StopIteration:
                    file_streams[i] = self.stream_file(self.get_next_file())
                    frames, actions_bin, is_first = next(file_streams[i])

                batch_frames.append(frames)
                batch_actions_bin.append(actions_bin)
                batch_are_first.append(is_first)

            yield (
                np.stack(batch_frames),
                np.stack(batch_actions_bin),
                np.stack(batch_are_first),
            )


def preprocess_inputs(
    frames: Tensor,
    actions_bin: Tensor,
    are_first: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize frames to [0, 1] and zero out hidden states on episode reset.

    Args:
        frames: [N, T, H, W, C]
        actions_bin: [N, T, 4]
        are_first: [N]
        hidden: [N, 1, D]

    Returns:
        Tuple of (frames_norm [N, T, H, W, C], target_actions_bin [N, T, 4], masked_hidden [N, 1, D]).
    """
    keep_hidden = (~are_first).to(DEVICE, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
    hidden_state = hidden * keep_hidden  # [N, 1, D]
    frames_gpu = frames.to(DEVICE, non_blocking=True)  # Compact uint8 host-to-device transfer
    frames_norm = frames_gpu.to(dtype=torch.float32).mul_(1.0 / 255.0)  # [N, T, H, W, C]
    target_actions_bin = actions_bin.to(DEVICE, dtype=torch.long)  # [N, T, 4]
    return frames_norm, target_actions_bin, hidden_state


def process_batch(
    model: Model,
    frames: Tensor,
    actions_bin: Tensor,
    are_first: Tensor,
    hidden: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pass batch through model and compute loss + entropy in float32.

    Args:
        frames: [N, T, H, W, C]
        actions_bin: [N, T, 4]
        are_first: [N]
        hidden: [N, 1, D]

    Returns:
        Tuple of (loss, hidden [N, 1, D], entropy, logits [N, T, 2], target_60 [N, T]).
    """
    frames_norm, target_actions_bin, hidden_state = preprocess_inputs(
        frames=frames,
        actions_bin=actions_bin,
        are_first=are_first,
        hidden=hidden,
    )
    logits, hidden_state = model(frames_norm, hidden_state)  # logits: [N, T, 2], hidden_state: [N, 1, D]
    hidden_state = hidden_state.detach()

    # --- Phase 1a: 60Hz Coarse Mode with Soft-WTA ---
    target_60 = target_actions_bin.max(dim=-1)[0]  # [N, T]
    K, N, _, _ = logits.shape

    head_losses = torch.stack(
        [
            F.cross_entropy(logits[k].transpose(1, 2), target_60, weight=CLASS_WEIGHTS, reduction="none").mean(dim=1)
            for k in range(K)
        ]
    )  # [K, N]

    tau = CONFIG["training"]["wtaTau"]
    weights = F.softmax(-head_losses / tau, dim=0).detach()  # [K, N]
    loss = (weights * head_losses).sum(dim=0).mean()

    best_head_idx = torch.argmin(head_losses, dim=0)  # [N]
    best_logits = torch.stack([logits[best_head_idx[n], n] for n in range(N)])  # [N, T, 2]

    probs = F.softmax(best_logits, dim=-1)  # [N, T, 2]
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

    # --- Phase 1b: 240Hz Subtick Control Mode ---
    # N, T, _ = logits.shape
    # logits_240 = logits.view(N, T * 4, 2)
    # target_actions_bin_240 = target_actions_bin.view(N, T * 4)
    # weights_tensor = torch.tensor(CONFIG["training"]["classWeights"], device=logits.device)
    # loss = F.cross_entropy(logits_240.transpose(1, 2), target_actions_bin_240, weight=weights_tensor)
    # probs = F.softmax(logits_240, dim=-1)
    # entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean()

    return loss, hidden_state, entropy, best_logits, target_60


def prepare_data_files() -> tuple[list[Path], list[Path], int, int, int]:
    """Scan training and validation dataset directory files and compute steps per epoch.

    Returns:
        Tuple of (train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch).
    """
    seq_len = CONFIG["training"]["seqLen"]
    batch_size = CONFIG["training"]["batchSize"]
    accumulation_steps = CONFIG["training"]["accumulationSteps"]

    train_dir = PROJECT_ROOT / "data" / "training"
    val_dir = PROJECT_ROOT / "data" / "validation"

    train_files = list(train_dir.glob("*.h5"))
    val_files = list(val_dir.glob("*.h5"))

    def get_chunks(f: Path) -> int:
        with h5py.File(f, "r") as data:
            return len(data["actions_bin"]) // seq_len

    total_train_chunks = sum(get_chunks(f) for f in train_files)
    train_steps_per_epoch = max(1, total_train_chunks // batch_size)

    total_val_chunks = sum(get_chunks(f) for f in val_files)
    val_steps_per_epoch = max(1, total_val_chunks // batch_size)

    opt_steps_per_epoch = (train_steps_per_epoch + accumulation_steps - 1) // accumulation_steps
    return train_files, val_files, train_steps_per_epoch, val_steps_per_epoch, opt_steps_per_epoch


def get_wsd_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> LambdaLR:
    """Warmup-Stable-Decay Learning Rate Scheduler configured from CONFIG."""
    warmup_ratio = CONFIG["training"]["warmupRatio"]
    decay_ratio = CONFIG["training"]["decayRatio"]
    min_lr_ratio = CONFIG["training"]["minLrRatio"]

    warmup_steps = int(total_steps * warmup_ratio)
    decay_steps = int(total_steps * decay_ratio)
    stable_steps = total_steps - warmup_steps - decay_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_step = step - (warmup_steps + stable_steps)
            progress = decay_step / decay_steps
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def init_model_and_optimizer(
    opt_steps_per_epoch: int,
) -> tuple[Model, torch.optim.Optimizer, LambdaLR]:
    """Instantiate Model, Adam optimizer, and Warmup-Stable-Decay (WSD) scheduler.

    Returns:
        Tuple of (model, optimizer, scheduler).
    """
    model = Model().to(DEVICE)
    lr = CONFIG["training"]["learningRate"]
    total_steps = CONFIG["training"]["epochs"] * opt_steps_per_epoch

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_wsd_scheduler(optimizer, total_steps=total_steps)
    return model, optimizer, scheduler


def load_checkpoint(checkpoint_file: str, state: dict) -> int:
    """Load model weights and training state if specified.

    Returns:
        Starting epoch number.
    """
    if not checkpoint_file:
        return 1

    checkpoint = torch.load(Path(checkpoint_file), map_location=DEVICE)
    state["model"].load_state_dict(checkpoint["model_state"])
    state["optimizer"].load_state_dict(checkpoint["optimizer_state"])
    state["scheduler"].load_state_dict(checkpoint["scheduler_state"])

    print(f"Loaded checkpoint {checkpoint_file}.")
    return checkpoint["epoch"] + 1


def log_diagnostics(
    state: dict,
    stats: dict,
    grad_rms: dict[str, float],
    update_ratios: dict[str, float],
):
    """Log training metrics, weight/gradient RMS values, and true Adam update ratios to WandB."""
    stats["global_step"] = state["global_step"]
    opt_steps = state["opt_steps"]
    stats["epoch"] = state["global_step"] / opt_steps
    model = state["model"]

    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name:
                continue
            parts = name.rsplit(".", 1)
            layer = parts[0] if len(parts) == 2 else name
            w_rms = (param.norm() / (param.numel() ** 0.5)).item()
            stats[f"weight_rms/{layer}"] = w_rms
            if name in grad_rms:
                stats[f"grad_rms/{layer}"] = grad_rms[name]
            if name in update_ratios:
                stats[f"update_ratio/{layer}"] = update_ratios[name]
    wandb.log(stats)


def optimize_and_evaluate(state: dict, loss: Tensor, entropy: Tensor):
    """Step optimizer/scheduler, clip gradients, and run periodic evaluation."""
    eval_freq = CONFIG["training"]["evalFreqSteps"]
    max_grad_norm = CONFIG["training"]["maxGradNorm"]

    model, optimizer, scheduler = state["model"], state["optimizer"], state["scheduler"]
    is_eval_step = (state["global_step"] + 1) % eval_freq == 0
    grad_rms = {}
    old_params = {}
    update_ratios = {}

    if is_eval_step:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "bias" in name:
                    continue
                if param.grad is not None:
                    grad_rms[name] = (param.grad.norm() / (param.grad.numel() ** 0.5)).item()
                old_params[name] = param.detach().clone()

    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item()
    optimizer.step()
    scheduler.step()

    if is_eval_step:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in old_params:
                    delta = param - old_params[name]
                    delta_rms = (delta.norm() / (delta.numel() ** 0.5)).item()
                    w_rms = (param.norm() / (param.numel() ** 0.5)).item()
                    if w_rms > 0:
                        update_ratios[name] = delta_rms / w_rms

    optimizer.zero_grad(set_to_none=True)
    state["global_step"] += 1

    if is_eval_step:
        val_loss, val_entropy, val_prec, val_rec, val_f1, val_latency = run_val(state)
        log_diagnostics(
            state=state,
            stats={
                "loss/train": loss.item(),
                "entropy/train": entropy.item(),
                "lr": scheduler.get_last_lr()[0],
                "loss/val": val_loss,
                "entropy/val": val_entropy,
                "val/precision": val_prec,
                "val/recall": val_rec,
                "val/f1": val_f1,
                "total_grad_norm": total_grad_norm,
                "inf_latency_ms": val_latency,
            },
            grad_rms=grad_rms,
            update_ratios=update_ratios,
        )
        model.train()


def run_train_epoch(state: dict, steps: int) -> float:
    """Run one training epoch over step count and return average train loss."""
    model, train_iter = state["model"], state["train_iter"]
    accum_steps = CONFIG["training"]["accumulationSteps"]

    model.train()
    num_batches = 0
    train_loss_tensor = torch.zeros(1, device=DEVICE)
    loss, entropy = None, None

    for i in range(steps):
        try:
            frames, actions_bin, are_first = next(train_iter)
        except StopIteration:
            break
        num_batches = i + 1

        loss, state["train_hidden"], entropy, _, _ = process_batch(
            model=model,
            frames=frames,
            actions_bin=actions_bin,
            are_first=are_first,
            hidden=state["train_hidden"],
        )

        (loss / accum_steps).backward()

        if num_batches % accum_steps == 0:
            optimize_and_evaluate(state, loss, entropy)

        train_loss_tensor += loss.detach()

    if num_batches % accum_steps != 0 and loss is not None:
        optimize_and_evaluate(state, loss, entropy)

    return (train_loss_tensor.item() / num_batches) if num_batches > 0 else 0.0


def run_val(state: dict) -> tuple[float, float, float, float, float, float]:
    """Run validation steps and measure inference latency and classification metrics.

    Returns:
        Tuple of (avg_val_loss, avg_val_entropy, precision, recall, f1, inference_latency_ms).
    """
    val_steps = CONFIG["training"]["valSteps"]
    model, val_iter = state["model"], state["val_iter"]

    model.eval()
    num_batches = 0
    val_loss_tensor = torch.zeros(1, device=DEVICE)
    val_entropy_tensor = torch.zeros(1, device=DEVICE)

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    with torch.no_grad():
        for i in range(val_steps):
            try:
                frames, actions_bin, are_first = next(val_iter)
            except StopIteration:
                break
            num_batches = i + 1

            loss, state["val_hidden"], entropy, logits, target_60 = process_batch(
                model=model,
                frames=frames,
                actions_bin=actions_bin,
                are_first=are_first,
                hidden=state["val_hidden"],
            )
            val_loss_tensor += loss.detach()
            val_entropy_tensor += entropy.detach()

            preds = logits.argmax(dim=-1)  # [N, T]
            total_tp += ((preds == 1) & (target_60 == 1)).sum().item()
            total_fp += ((preds == 1) & (target_60 == 0)).sum().item()
            total_fn += ((preds == 0) & (target_60 == 1)).sum().item()

        dummy_x = torch.zeros(1, 1, 480, 640, 3, device=DEVICE)
        dummy_h = torch.zeros(1, 1, model.hidden_dim, device=DEVICE)
        model(dummy_x, dummy_h)
        torch.mps.synchronize()

        t0 = time.perf_counter()
        model(dummy_x, dummy_h)
        torch.mps.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

    avg_val_loss = (val_loss_tensor.item() / num_batches) if num_batches > 0 else 0.0
    avg_val_entropy = (val_entropy_tensor.item() / num_batches) if num_batches > 0 else 0.0

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn)
    f1 = (2 * precision * recall) / (precision + recall + 1e-9)

    return avg_val_loss, avg_val_entropy, precision, recall, f1, elapsed_ms


def save_checkpoint(epoch: int, state: dict, train_loss: float, val_loss: float | None = None):
    """Save epoch model, optimizer, and scheduler state to checkpoint file."""
    checkpoint_data = {
        "epoch": epoch,
        "model_state": state["model"].state_dict(),
        "optimizer_state": state["optimizer"].state_dict(),
        "scheduler_state": state["scheduler"].state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_path = checkpoints_dir / f"epoch_{epoch}.pt"
    torch.save(checkpoint_data, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def train():
    cfg_tr = CONFIG["training"]
    batch_size = cfg_tr["batchSize"]
    epochs = cfg_tr["epochs"]

    train_files, val_files, train_steps, _, opt_steps = prepare_data_files()

    train_loader = DataLoader(DatasetGenerator(train_files, is_val=False), batch_size=None)
    val_loader = DataLoader(DatasetGenerator(val_files, is_val=True), batch_size=None)

    model, optimizer, scheduler = init_model_and_optimizer(opt_steps)
    hidden_dim = CONFIG["model"]["hiddenDim"]

    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_iter": iter(train_loader),
        "val_iter": iter(val_loader),
        "global_step": 0,
        "opt_steps": opt_steps,
        "train_hidden": torch.zeros(batch_size, 1, hidden_dim, device=DEVICE),
        "val_hidden": torch.zeros(batch_size, 1, hidden_dim, device=DEVICE),
    }

    start_epoch = load_checkpoint(CONFIG["checkpointFile"], state)
    state["global_step"] = (start_epoch - 1) * opt_steps

    wandb.init(
        project="ai-doggie",
        name="more epochs + wsd scheduler",
        config=cfg_tr,
    )
    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("*", step_metric="epoch")

    for epoch in range(start_epoch, epochs + 1):
        avg_train_loss = run_train_epoch(state, train_steps)

        if epoch % 5 == 0 or epoch == epochs:
            save_checkpoint(epoch, state, avg_train_loss)

    wandb.finish()


if __name__ == "__main__":
    train()
