"""Contains model inference procedure and allows shutdown via pressing the esc key.

Example:
    $ python infer.py
"""

import json
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from struct import pack, unpack

import numpy as np
import torch
from pynput.keyboard import Key, Listener
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model

with (Path(__file__).resolve().parent / "config.json").open() as f:
    _CONFIG = json.load(f)

_DEVICE = torch.device("mps")

_is_inferring = False
_is_shutdown = False


def _on_press(key):
    global _is_shutdown
    global _is_inferring

    if key == Key[_CONFIG["keys"]["exitKeyName"]]:
        _is_shutdown = True
    elif key == Key[_CONFIG["keys"]["recordKeyName"]]:
        _is_inferring = True


def _init_shm() -> SharedMemory:
    """Initialize a shared memory block between this script and the c++ mod."""
    try:
        shm = SharedMemory(name="GDMem")
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

    shm = SharedMemory(
        name="GDMem",
        create=True,
        size=921616,
    )
    shm.buf[0:16] = bytes(16)
    return shm


def _init_model() -> Model:
    """Load model and weights, placing it on the MPS device in evaluation mode."""
    model = Model().to(_DEVICE)

    checkpoint_file = _CONFIG["checkpointFile"]
    if checkpoint_file:
        checkpoint = torch.load(
            (Path(__file__).resolve().parents[1] / "checkpoints") / checkpoint_file, map_location=_DEVICE
        )
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loading checkpoint {checkpoint_file}.")

    model.eval()
    return model


def _run_inference_loop(model: Model, shm: SharedMemory):
    """Run the main loop retrieving frames, performing inference, and logging stats."""
    hidden_state: Tensor = torch.zeros(  # [N, L, D]
        1,
        1,
        _CONFIG["model"]["hiddenDim"],
        device=_DEVICE,
    )

    i = 0
    last_tick = -1

    with torch.inference_mode():
        while not _is_shutdown:
            if not _is_inferring:
                continue

            # Wait for C++ to set frameReadyBin == 1
            if unpack("i", shm.buf[8:12])[0] != 1:
                time.sleep(0)
                continue

            i += 1
            time_start: float = time.perf_counter()

            # Read current game tick and dimensions
            current_tick = unpack("i", shm.buf[0:4])[0]
            if current_tick == last_tick:
                shm.buf[12:16] = pack("i", 1)
                shm.buf[8:12] = pack("i", 0)
                continue

            if current_tick < last_tick:
                print(f"\nDeath detected (tick: {last_tick} -> {current_tick})! Resetting hidden state.")
                hidden_state = torch.zeros(
                    1,
                    1,
                    _CONFIG["model"]["hiddenDim"],
                    device=_DEVICE,
                )
            last_tick = current_tick

            frame_w = _CONFIG["frame"]["width"]
            frame_h = _CONFIG["frame"]["height"]

            # logits: [N, T, V]
            # hidden_state: [N, L, D]
            logits, hidden_state = model(
                torch.from_numpy(
                    np.frombuffer(
                        shm.buf[16 : 16 + frame_w * frame_h * 3],
                        dtype=np.uint8,
                    ).reshape((frame_h, frame_w, 3))
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=_DEVICE, dtype=torch.float32)
                / 255.0,
                hidden_state,
            )

            actions = torch.argmax(logits.view(4, 2), dim=-1).cpu().tolist()

            # Write actions to currActionBin
            shm.buf[4:8] = pack("i", actions[0] | (actions[1] << 1) | (actions[2] << 2) | (actions[3] << 3))

            # Handshake acknowledgement: set actionReadyBin = 1, frameReadyBin = 0
            shm.buf[12:16] = pack("i", 1)
            shm.buf[8:12] = pack("i", 0)

            if i % (_CONFIG["logIntervalSec"] * _CONFIG["fps"]) == 0:
                print(f"\rInference latency: {(time.perf_counter() - time_start) * 1000:.2f}ms", end="", flush=True)


def _infer():
    """Coordinate resource initialization, launch background threads, and start loop."""
    Listener(on_press=_on_press).start()

    shm = _init_shm()

    try:
        _run_inference_loop(_init_model(), shm)
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    _infer()
