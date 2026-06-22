"""Contains model inference procedure and allows shutdown via pressing the esc key.

Example:
    $ python infer.py
"""

import json
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from struct import pack

import torch
from pynput.keyboard import Key, Listener
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.model import Model
from game.game_env import GameEnv

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
    _CONFIG = json.load(f)

_is_inferring = False
_is_shutdown = False
_curr_action_bin = 0


def _on_press(key):
    global _is_shutdown
    global _is_inferring

    if key == Key[_CONFIG["keys"]["exitKeyName"]]:
        _is_shutdown = True
    elif key == Key[_CONFIG["keys"]["recordKeyName"]]:
        _is_inferring = True


def _init_shm():
    """Initialize a shared memory block between this script and the c++ mod."""
    global _curr_action_bin

    shm_name = _CONFIG["shmName"]
    try:
        shm = SharedMemory(name=shm_name)
        shm.buf[0:16] = bytes(16)
    except FileNotFoundError:
        shm: SharedMemory = SharedMemory(
            name=shm_name,
            create=True,
            size=16,
        )

    _curr_action_bin = 0

    # -1 means inference mode.
    shm.buf[12:16] = pack("i", -1)

    while not _is_shutdown:
        # 4:8 - Current action.
        shm.buf[4:8] = pack("i", _curr_action_bin)
        time.sleep(0)

    shm.close()
    shm.unlink()


def _infer():
    """Load model and weights, track frame drops and latency, and process model output."""
    listener = Listener(on_press=_on_press)
    listener.start()

    shm_thread = threading.Thread(target=_init_shm, daemon=True)
    shm_thread.start()

    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    device: torch.device = torch.device(_CONFIG["model"]["deviceName"])

    base_dir = Path(__file__).resolve().parents[2]
    checkpoint_dir: Path = base_dir / _CONFIG["fileNames"]["checkpointDirName"]
    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]

    model: Model = Model().to(device)

    if checkpoint_name:
        checkpoint: dict[str, int | float | dict[str, int | Tensor]] = torch.load(
            checkpoint_dir / checkpoint_name, map_location=device
        )

        model.load_state_dict(checkpoint["model_state"])
        print(f"Loading checkpoint {checkpoint_name}.")

    model.eval()

    model = torch.compile(model)

    env: GameEnv = GameEnv()

    hidden_state: Tensor = torch.zeros(  # [N, L, D]
        1,
        1,
        hidden_state_dim,
        device=device,
    )
    curr_action_bin = 0

    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]
    now = time.perf_counter()
    frame_drop_cache = env.capture_engine.frame_drops

    i = 0
    with torch.inference_mode():
        while not _is_shutdown:
            if not _is_inferring:
                continue

            i += 1

            frame_HWC, _ = env.get_frame()

            time_start: float = time.perf_counter()

            frame_NTHWC = torch.from_numpy(frame_HWC).unsqueeze(0).unsqueeze(0)
            frame_NTHWC = frame_NTHWC.to(device=device, dtype=torch.float32) / 255

            logits: Tensor  # [N, T, V]
            hidden_state: Tensor  # [N, L, D]
            logits, hidden_state = model(frame_NTHWC, hidden_state)

            curr_action_bin = torch.argmax(logits, dim=-1).item()
            global _curr_action_bin
            _curr_action_bin = curr_action_bin

            infer_time: float = (time.perf_counter() - time_start) * 1000

            if i % log_interval == 0:
                elapsed = time.perf_counter() - now
                drops = env.capture_engine.frame_drops - frame_drop_cache
                print(
                    f"\rInference latency: {infer_time:.2f}ms | Frame drops: {drops / elapsed:.2f}/s",
                    end="",
                    flush=True,
                )
                now, frame_drop_cache = (
                    time.perf_counter(),
                    env.capture_engine.frame_drops,
                )

    env.capture_engine.stop_capture_stream()


if __name__ == "__main__":
    _infer()
