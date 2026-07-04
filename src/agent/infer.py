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
from game.screen_capture import start_capture_engine

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
        time.sleep(_CONFIG["recordStartDelaySec"])
        _is_inferring = True


def _init_shm() -> SharedMemory:
    """Initialize a shared memory block between this script and the c++ mod."""
    global _curr_action_bin

    shm_name = _CONFIG["shmName"]
    try:
        shm = SharedMemory(name=shm_name)
        shm.buf[0:16] = bytes(16)
    except FileNotFoundError:
        shm = SharedMemory(
            name=shm_name,
            create=True,
            size=16,
        )

    _curr_action_bin = 0

    # -1 means inference mode.
    shm.buf[12:16] = pack("i", -1)
    return shm


def _active_shm_process(shm: SharedMemory):
    """Periodically update the current action in the shared memory block."""
    while not _is_shutdown:
        # 4:8 - Current action.
        shm.buf[4:8] = pack("i", _curr_action_bin)
        time.sleep(0)

    shm.close()
    shm.unlink()


def _init_model() -> Model:
    """Load model and weights, placing it on the MPS device in evaluation mode."""
    device = torch.device("mps")
    model = Model().to(device)

    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]
    if checkpoint_name:
        checkpoint_dir = Path(__file__).resolve().parents[2] / _CONFIG["fileNames"]["checkpointDirName"]
        checkpoint = torch.load(checkpoint_dir / checkpoint_name, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loading checkpoint {checkpoint_name}.")

    model.eval()
    return model


def _preprocess_frame(bgra_frame) -> Tensor:
    """Preprocess the BGRA frame into a normalized PyTorch tensor [N, T, H, W, C]."""

    # H, W, _ = frame_HWC.shape
    # input_H = _CONFIG["model"]["inputHeightPx"]
    # input_W = _CONFIG["model"]["inputWidthPx"]

    # h_offset = (H - input_H) // 2 if H > input_H else 0
    # w_offset = (W - input_W) // 2 if W > input_W else 0
    # frame_HWC = frame_HWC[h_offset : h_offset + input_H, w_offset : w_offset + input_W, :]

    frame_HWC = bgra_frame[:, :, :3].copy()
    frame_NTHWC = torch.from_numpy(frame_HWC).unsqueeze(0).unsqueeze(0)
    return frame_NTHWC.to(device=torch.device("mps"), dtype=torch.float32) / 255


def _run_inference_loop(model: Model, capture_engine):
    """Run the main loop retrieving frames, performing inference, and logging stats."""
    global _curr_action_bin

    device = torch.device("mps")
    hidden_state: Tensor = torch.zeros(  # [N, L, D]
        1,
        1,
        _CONFIG["model"]["hiddenDim"],
        device=device,
    )

    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]
    now = time.perf_counter()
    frame_drop_cache = capture_engine.frame_drops
    i = 0

    with torch.inference_mode():
        while not _is_shutdown:
            if not _is_inferring:
                continue

            i += 1

            bgra_frame = capture_engine.queue_full.get()
            frame_NTHWC = _preprocess_frame(bgra_frame)
            capture_engine.queue_empty.put_nowait(bgra_frame)

            time_start: float = time.perf_counter()

            # logits: Tensor  # [N, T, V]
            # hidden_state: Tensor  # [N, L, D]
            logits, hidden_state = model(frame_NTHWC, hidden_state)

            _curr_action_bin = torch.argmax(logits, dim=-1).item()

            infer_time: float = (time.perf_counter() - time_start) * 1000

            if i % log_interval == 0:
                elapsed = time.perf_counter() - now
                drops = capture_engine.frame_drops - frame_drop_cache
                print(
                    f"\rInference latency: {infer_time:.2f}ms | Frame drops: {drops / elapsed:.2f}/s",
                    end="",
                    flush=True,
                )
                now, frame_drop_cache = (
                    time.perf_counter(),
                    capture_engine.frame_drops,
                )


def _infer():
    """Coordinate resource initialization, launch background threads, and start loop."""
    listener = Listener(on_press=_on_press)
    listener.start()

    shm = _init_shm()
    shm_thread = threading.Thread(target=_active_shm_process, args=(shm,), daemon=True)
    shm_thread.start()

    model = _init_model()
    capture_engine = start_capture_engine()

    try:
        _run_inference_loop(model, capture_engine)
    finally:
        capture_engine.stop_capture_stream()


if __name__ == "__main__":
    _infer()
