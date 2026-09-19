import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

DEVICE = torch.device("mps")
MAX_HORIZON = CONFIG["training"]["seqLen"]


def load_dataset(h5_files: list[Path], max_horizon: int) -> TensorDataset:
    all_frames: list[Tensor] = []
    all_states: list[Tensor] = []
    all_actions: list[Tensor] = []
    all_targets: list[Tensor] = []

    for fpath in h5_files:
        with h5py.File(fpath, "r") as f:
            for grp in f["rollouts"].values():
                frame = torch.from_numpy(grp["frame_0"][:]).permute(2, 0, 1)
                state = torch.from_numpy(grp["aux_state"][:])

                act = torch.from_numpy(grp["actions"][:max_horizon]).float()
                if len(act) < max_horizon:
                    act = F.pad(act, (0, max_horizon - len(act)), value=-1.0)

                all_frames.append(frame)
                all_states.append(state)
                all_actions.append(act)
                all_targets.append(torch.tensor(float(grp.attrs["ftd"]), dtype=torch.float32))

    if not all_frames:
        raise ValueError(f"No rollouts found in provided files: {h5_files}")

    frames_tensor = torch.stack(all_frames)
    states_tensor = torch.stack(all_states)
    actions_tensor = torch.stack(all_actions)
    targets_tensor = torch.stack(all_targets)

    print(f"Loaded {len(frames_tensor)} rollouts into in-memory TensorDataset.")
    return TensorDataset(frames_tensor, states_tensor, actions_tensor, targets_tensor)


def get_lr_lambda(
    step: int,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    if step < (total_steps - decay_steps):
        return 1.0
    progress = float(step - (total_steps - decay_steps)) / float(max(1, decay_steps))
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def log_diagnostics(
    model: Model,
    stats: dict,
    grad_rms: dict[str, float],
    update_ratios: dict[str, float],
):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad or "weight" not in name:
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


def measure_inference_latency(model: Model) -> float:
    dummy_x = torch.zeros(1, 3, 480, 640, device=DEVICE)
    dummy_s = torch.zeros(1, 4, device=DEVICE)
    dummy_a = torch.zeros(1, MAX_HORIZON, device=DEVICE)
    with torch.no_grad():
        model(dummy_x, dummy_s, dummy_a)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            model(dummy_x, dummy_s, dummy_a)
        torch.mps.synchronize()
        elapsed_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
    return elapsed_ms


def run_val(model: Model, val_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for frames, states, actions, targets in val_loader:
            frames = frames.to(DEVICE).float() / 255.0
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(frames, states, actions).squeeze(-1)
            val_loss += F.smooth_l1_loss(preds, targets).item()

    avg_val_loss = val_loss / len(val_loader)
    inf_latency = measure_inference_latency(model)
    return avg_val_loss, inf_latency


def main():
    train_dir = PROJECT_ROOT / "data" / "train"
    val_dir = PROJECT_ROOT / "data" / "val"
    train_files = sorted(list(train_dir.glob("*.h5")))
    val_files = sorted(list(val_dir.glob("*.h5")))

    train_dataset = load_dataset(train_files, max_horizon=MAX_HORIZON)
    val_dataset = load_dataset(val_files, max_horizon=MAX_HORIZON)

    batch_size = CONFIG["training"]["batchSize"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["training"]["learningRate"],
        weight_decay=CONFIG["training"]["weightDecay"],
    )

    epochs = CONFIG["training"]["epochs"]
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * CONFIG["training"]["warmupRatio"])
    decay_steps = int(total_steps * CONFIG["training"]["decayRatio"])
    min_lr_ratio = CONFIG["training"]["minLrRatio"]
    eval_freq = CONFIG["training"]["evalFreqSteps"]

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: get_lr_lambda(s, total_steps, warmup_steps, decay_steps, min_lr_ratio),
    )

    wandb.init(
        project="ai-doggie",
        name=f"ftd training baby",
        config=CONFIG["training"],
    )
    wandb.define_metric("global_step", hidden=True)
    wandb.define_metric("*", step_metric="global_step")

    global_step = 0
    last_train_loss = 0.0

    for _ in range(epochs):
        model.train()

        for frames, states, actions, targets in train_loader:
            global_step += 1

            frames = frames.to(DEVICE).float() / 255.0
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)
            targets = targets.to(DEVICE)

            preds = model(frames, states, actions).squeeze(-1)
            loss = F.smooth_l1_loss(preds, targets)
            last_train_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()

            is_eval_step = (global_step % eval_freq == 0)
            grad_rms = {}
            old_params = {}

            if is_eval_step:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if "bias" in name:
                            continue
                        if param.grad is not None:
                            grad_rms[name] = (param.grad.norm() / (param.grad.numel() ** 0.5)).item()
                        old_params[name] = param.detach().clone()

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CONFIG["training"]["maxGradNorm"],
            ).item()

            optimizer.step()
            scheduler.step()

            if is_eval_step:
                update_ratios = {}
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in old_params:
                            delta = param - old_params[name]
                            delta_rms = (delta.norm() / (delta.numel() ** 0.5)).item()
                            w_rms = (param.norm() / (param.numel() ** 0.5)).item()
                            if w_rms > 0:
                                update_ratios[name] = delta_rms / w_rms

                val_loss, inf_latency = run_val(model, val_loader)
                stats = {
                    "train/step_loss": last_train_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "total_grad_norm": total_grad_norm,
                    "val/loss": val_loss,
                    "inf_latency_ms": inf_latency,
                    "global_step": global_step,
                }
                log_diagnostics(model, stats, grad_rms, update_ratios)
                model.train()

    # Save model
    final_val_loss, _ = run_val(model, val_loader)
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"model_pretrain_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    checkpoint_data = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "train_loss": last_train_loss,
        "val_loss": final_val_loss,
    }
    torch.save(checkpoint_data, save_path)
    print(f"\nModel checkpoint saved successfully to {save_path}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
