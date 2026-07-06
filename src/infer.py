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

_is_inferring = False
_is_shutdown = False


def _on_press(key):
    global _is_shutdown
    global _is_inferring

    if key == Key[_CONFIG["keys"]["exitKeyName"]]:
        _is_shutdown = True
    elif key == Key[_CONFIG["keys"]["recordKeyName"]]:
        time.sleep(_CONFIG["recordStartDelaySec"])
        _is_inferring = True


def _init_shm() -> SharedMemory:
    """Initialize a shared memory block between this script and the c++ mod."""
    shm_name = _CONFIG["shmName"]
    try:
        shm = SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

    shm = SharedMemory(
        name=shm_name,
        create=True,
        size=6912024,
    )
    shm.buf[0:24] = bytes(24)
    return shm


def _init_model() -> Model:
    """Load model and weights, placing it on the MPS device in evaluation mode."""
    device = torch.device("mps")
    model = Model().to(device)

    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]
    if checkpoint_name:
        checkpoint_dir = Path(__file__).resolve().parents[1] / _CONFIG["fileNames"]["checkpointDirName"]
        checkpoint = torch.load(checkpoint_dir / checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loading checkpoint {checkpoint_name}.")

    model.eval()
    return model


def _run_inference_loop(model: Model, shm: SharedMemory):
    """Run the main loop retrieving frames, performing inference, and logging stats."""
    device = torch.device("mps")
    hidden_state: Tensor = torch.zeros(  # [N, L, D]
        1,
        1,
        _CONFIG["model"]["hiddenDim"],
        device=device,
    )

    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]
    i = 0

    with torch.inference_mode():
        while not _is_shutdown:
            if not _is_inferring:
                continue

            # Wait for C++ to set frameReadyBin == 1
            frame_ready = unpack("i", shm.buf[8:12])[0]
            if frame_ready != 1:
                time.sleep(0.001)
                continue

            i += 1
            time_start: float = time.perf_counter()

            # Read current game dimensions
            width = unpack("i", shm.buf[16:20])[0]
            height = unpack("i", shm.buf[20:24])[0]

            # Extract raw frame from shared memory
            frame_size = width * height * 3
            raw_frame_rgb = np.frombuffer(shm.buf[24 : 24 + frame_size], dtype=np.uint8).reshape((height, width, 3))

            # Convert to [1, 1, H, W, C] RGB tensor on MPS
            frame_NTHWC = (
                torch.from_numpy(raw_frame_rgb)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=torch.device("mps"), dtype=torch.float32)
                / 255.0
            )

            # logits: [N, T, V]
            # hidden_state: [N, L, D]
            logits, hidden_state = model(frame_NTHWC, hidden_state)

            actions = torch.argmax(logits.view(4, 2), dim=-1).cpu().tolist()

            # Pack the 4 actions into a bitmask
            curr_action_bin = actions[0] | (actions[1] << 1) | (actions[2] << 2) | (actions[3] << 3)

            # Write actions to currActionBin
            shm.buf[4:8] = pack("i", curr_action_bin)

            infer_time: float = (time.perf_counter() - time_start) * 1000

            # Handshake acknowledgement: set actionReadyBin = 1, frameReadyBin = 0
            shm.buf[12:16] = pack("i", 1)
            shm.buf[8:12] = pack("i", 0)

            if i % log_interval == 0:
                print(f"\rInference latency: {infer_time:.2f}ms", end="", flush=True)


def _infer():
    """Coordinate resource initialization, launch background threads, and start loop."""
    listener = Listener(on_press=_on_press)
    listener.start()

    shm = _init_shm()
    model = _init_model()

    try:
        _run_inference_loop(model, shm)
    finally:
        shm.close()


if __name__ == "__main__":
    _infer()
