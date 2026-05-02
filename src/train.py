"""Handles logic from loading the saved data files, to loading the previous checkpoint and continuing training.

Example:
    $ python train.py
"""

import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32, Int64, UInt8
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from config import CONFIG as _CONFIG
from policy_model import PolicyModel

_CONFIG_TRAINING = _CONFIG["training"]

BATCH_SIZE: int = _CONFIG_TRAINING["batchSize"]
LR = _CONFIG_TRAINING["learningRate"]


class _DatasetGenerator(IterableDataset):
    """Yield training batches built from parallel gameplay streams."""

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Batch together mini batches from each file stream.

        Yields:
            Tuple of arrays of frames, action binaries, and is_first flags of the concatenated mini batches.
        """
        dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
        dataset_files_src: Path = Path(__file__).resolve().parents[1] / dataset_dir_name

        dataset_files: list[Path] = list(dataset_files_src.glob("*.npz"))
        random.shuffle(dataset_files)

        if len(dataset_files) < BATCH_SIZE:
            err_msg = "Not enough data files."
            raise Exception(err_msg)  # noqa: TRY002

        file_streams: list[Iterator[tuple[np.ndarray, np.ndarray, bool]]] = [
            self._stream_file(dataset_files.pop(0)) for _ in range(BATCH_SIZE)
        ]

        while True:
            active_stream_count = 0

            batch_frames: list[np.ndarray] = []
            batch_actions_bin: list[np.ndarray] = []
            batch_are_first: list[bool] = []

            for batch_idx, curr_file_stream in enumerate(file_streams):
                frames: UInt8[np.ndarray, "N H W C"]
                actions_bin: UInt8[np.ndarray, "N"]
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
        seq_len: int = _CONFIG_TRAINING["seqLen"]

        with np.load(filepath) as data:
            frames: np.ndarray
            actions_bin: np.ndarray
            frames, actions_bin = data["frames"], data["actions_bin"]

            num_chunks = len(frames) // seq_len

            # Chop up each file stream into chunks with length seq_len.
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * seq_len

                chunk_frames: np.ndarray = frames[start_idx : start_idx + seq_len]
                chunk_actions_bin: np.ndarray = actions_bin[
                    start_idx : start_idx + seq_len + 1
                ]

                yield chunk_frames, chunk_actions_bin, (chunk_idx == 0)


class _AdamW:
    """Manual implementation of the AdamW optimizer."""

    def __init__(self, params: Iterable[Tensor]):
        """Initialize the model parameters, first and second moments."""
        self._params: list[Tensor] = list(params)

        self._m: list[Tensor] = [torch.zeros_like(p) for p in self._params]
        self._v: list[Tensor] = [torch.zeros_like(p) for p in self._params]

        self.step_idx = 1

    # Ensure autograd graph doesn't get contaminated from tensor ops.
    @torch.no_grad()
    def step(self):
        """Perform a parameter update step."""
        beta1: float = _CONFIG_TRAINING["beta1"]
        beta2: float = _CONFIG_TRAINING["beta2"]
        W_decay: float = _CONFIG_TRAINING["weightDecay"]

        for i, param in enumerate(self._params):
            grad = param.grad
            if grad is None:
                err_msg = "Parameter has no gradient."
                raise Exception(err_msg)  # noqa: TRY002

            # Store the pre-corrected moments.
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * grad
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * (grad**2)

            # Correct bias introduced with 0-intialization.
            m_hat: Tensor = self._m[i] / (1 - beta1**self.step_idx)
            v_hat: Tensor = self._v[i] / (1 - beta2**self.step_idx)

            param_adam = self._params[i] - LR * m_hat / (torch.sqrt(v_hat) + 1e-8)

            # Apply decoupled weight decay.
            self._params[i].copy_(param_adam - LR * W_decay * param)

        self.step_idx += 1

    def clear_grad(self):
        for param in self._params:
            param.grad = None

    def get_state_dict(self) -> dict[str, int | list[Tensor]]:
        """Return the optimizer state.

        Returns:
            Dictionary of the step count, 1st, and 2nd moments.
        """
        return {
            "step_idx": self.step_idx,
            "mean": self._m,
            "var": self._v,
        }

    def load_state_dict(self, state_dict: dict[str, int | list[Tensor]]):
        self.step_idx = state_dict["step_idx"]
        self._m = state_dict["mean"]
        self._v = state_dict["var"]


def _softmax(logits: Float32[Tensor, "N V"]) -> Float32[Tensor, "N V"]:
    logits_exp_stable = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
    return logits_exp_stable / torch.sum(logits_exp_stable, dim=1, keepdim=True)


def _cross_entropy(
    logits: Float32[Tensor, "N V"],
    target: Int64[Tensor, "actions"],
) -> Tensor:
    batches = torch.arange(target.shape[0], device=target.device)
    return -torch.log(_softmax(logits)[batches, target]).mean()


def _train():
    """Load model, previous checkpoints, and dataset. Train over epochs hyper-parameter."""
    device: torch.device = torch.device(_CONFIG["model"]["deviceName"])
    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    epochs = _CONFIG["training"]["epochs"]
    checkpoint_save_interval: int = _CONFIG["training"]["checkpointSaveInterval"]

    dataloader: DataLoader = DataLoader(_DatasetGenerator(), batch_size=None)
    model: PolicyModel = PolicyModel().to(device)
    optimizer: _AdamW = _AdamW(model.parameters())

    checkpoint_dir = (
        Path(__file__).resolve().parents[1] / _CONFIG["fileNames"]["checkpointDirName"]
    )
    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]
    checkpoint: dict[str, int | float | dict[str, int | Tensor]] = torch.load(
        checkpoint_dir / checkpoint_name,
    )

    start_epoch: int = checkpoint["epoch"] + 1
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    for epoch in range(start_epoch, epochs + 1):
        epoch_loss: float = 0
        num_batches = 0

        hidden_state: Float32[torch.Tensor, "N L D"] = torch.zeros(
            BATCH_SIZE,
            1,
            hidden_state_dim,
            device=device,
        )

        for i, (frames, actions_bin, are_first) in enumerate(dataloader):
            # Note that the variables from dataloader are actually concatenated from different mini batches.
            frames: np.ndarray
            actions_bin: np.ndarray
            are_first: np.ndarray

            vocab_size: int = _CONFIG["model"]["vocabSize"]

            num_batches = i + 1

            keep_hidden_mask = ~are_first
            keep_hidden_mask: Float32[Tensor, "N L D"] = (
                torch.from_numpy(
                    keep_hidden_mask,
                )
                .to(
                    device,
                    dtype=torch.float32,
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            hidden_state = hidden_state * keep_hidden_mask

            frame_norm: Float32[Tensor, "N H W C"] = (
                torch.from_numpy(frames).to(device, dtype=torch.float32) / 255
            )
            target_actions_bin: Int64[Tensor, "N seq_len"] = torch.from_numpy(
                actions_bin[:, 1:],
            ).to(
                device,
                dtype=torch.long,
            )

            logits: Float32[Tensor, "N T V"]
            hidden_state: Float32[Tensor, "N L D"]
            logits, hidden_state = model(frame_norm, hidden_state)

            # Ensure hidden state doesn't effect the gradients of the entire dataset.
            hidden_state = hidden_state.detach()

            logits: Float32[Tensor, "N_nonsequential V"] = logits.reshape(
                -1,
                vocab_size,
            )
            target_actions: Int64[Tensor, "all_actions"] = target_actions_bin.reshape(
                -1,
            )

            loss: Tensor = _cross_entropy(
                logits=logits,
                target=target_actions,
            )
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} completed | Average loss: {avg_loss:.4f}")

        if epoch % checkpoint_save_interval == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.get_state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    _train()
