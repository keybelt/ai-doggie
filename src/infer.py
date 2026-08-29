import json
import sys
import time
from pathlib import Path

import torch
from multiprocessing.shared_memory import SharedMemory
from pynput.keyboard import Key, Listener

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model
from shm_utils import acknowledge_handshake, get_frame, init_shm, wait_for_next_frame

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

DEVICE = torch.device("mps")

is_shutdown = False


def init_model() -> Model:
    """Instantiate Model and load weights from checkpoint if configured.

    Returns:
        Model instance set to evaluation mode.
    """
    model = Model().to(DEVICE)
    checkpoint_file = CONFIG.get("checkpointFile")
    if checkpoint_file:
        checkpoints_dir = Path(__file__).resolve().parents[1] / "checkpoints"
        checkpoint_path = checkpoints_dir / checkpoint_file
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint {checkpoint_file}.")
    model.eval()
    return model


def run_inference_loop(model: Model, shm: SharedMemory):
    """Run real-time game frame capture, model forward pass, and action IPC output loop."""
    hidden_dim: int = CONFIG["model"]["hiddenDim"]
    frame_w: int = CONFIG["frame"]["width"]
    frame_h: int = CONFIG["frame"]["height"]
    log_interval: int = CONFIG["logIntervalSec"] * CONFIG["fps"]

    hidden_state = torch.zeros(1, 1, hidden_dim, device=DEVICE)
    i = 0
    last_tick = -1

    with torch.inference_mode():
        while not is_shutdown:
            current_tick, is_ready, _, _ = wait_for_next_frame(shm, last_tick)
            if not is_ready:
                continue

            i += 1
            time_start = time.perf_counter()

            if current_tick < last_tick:
                print(f"\nDeath detected (tick: {last_tick} -> {current_tick})! Resetting hidden state.")
                hidden_state = torch.zeros(1, 1, hidden_dim, device=DEVICE)

            last_tick = current_tick
            frame = get_frame(shm, frame_w, frame_h)
            frame_tensor = (
                torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.float32) / 255.0
            )

            q_values, hidden_state = model(frame_tensor, hidden_state)
            action_val = 1 if q_values[0, 0, 1] > q_values[0, 0, 0] else 0
            acknowledge_handshake(shm, action_val)

            if i % log_interval == 0:
                latency = (time.perf_counter() - time_start) * 1000
                print(f"\rInference latency: {latency:.2f}ms", end="", flush=True)


def infer():
    """Coordinate shared memory, launch keyboard listener, and run inference."""

    def on_press(key):
        global is_shutdown
        exit_key = Key[CONFIG["keys"]["exitKeyName"]]
        if key == exit_key:
            is_shutdown = True

    listener = Listener(on_press=on_press)
    listener.start()

    shm = init_shm()

    try:
        model = init_model()
        run_inference_loop(model, shm)
    finally:
        listener.stop()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    infer()
