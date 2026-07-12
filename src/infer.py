"""Contains model inference procedure and allows shutdown via pressing the esc key.

Example:
    $ python infer.py
"""

import json
import sys
import time
from pathlib import Path

import torch
from pynput.keyboard import Key, Listener
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parent))

from model import Model
from shm_utils import init_shm, get_frame, acknowledge_handshake, wait_for_next_frame


def _init_model(config: dict) -> Model:
    """Load model and weights, placing it on the MPS device in evaluation mode."""
    DEVICE = torch.device("mps")
    model = Model().to(DEVICE)

    checkpoint_file = config["checkpointFile"]
    if checkpoint_file:
        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
        checkpoint_path = CHECKPOINTS_DIR / checkpoint_file
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loading checkpoint {checkpoint_file}.")

    model.eval()
    return model


def _run_inference_loop(model: Model, shm, state: dict, config: dict):
    """Run the main loop retrieving frames, performing inference, and logging stats."""
    DEVICE = torch.device("mps")
    hidden_dim = config["model"]["hiddenDim"]
    frame_w = config["frame"]["width"]
    frame_h = config["frame"]["height"]
    log_interval = config["logIntervalSec"] * config["fps"]

    hidden_state: Tensor = torch.zeros(1, 1, hidden_dim, device=DEVICE)

    i = 0
    last_tick = -1

    with torch.inference_mode():
        while not state["is_shutdown"]:
            # Read header state and wait for next frame
            current_tick, is_ready = wait_for_next_frame(shm, last_tick)
            if not is_ready:
                continue

            i += 1
            time_start: float = time.perf_counter()

            if current_tick < last_tick:
                print(f"\nDeath detected (tick: {last_tick} -> {current_tick})! Resetting hidden state.")
                hidden_state = torch.zeros(1, 1, hidden_dim, device=DEVICE)

            last_tick = current_tick

            # Get and preprocess frame
            frame = get_frame(shm, frame_w, frame_h)
            frame_tensor = (
                torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device=DEVICE, dtype=torch.float32) / 255.0
            )

            # Perform inference
            logits, hidden_state = model(frame_tensor, hidden_state)
            actions = torch.argmax(logits.view(4, 2), dim=-1).cpu().tolist()

            # Pack action value and acknowledge handshake
            action_val = actions[0] | (actions[1] << 1) | (actions[2] << 2) | (actions[3] << 3)
            acknowledge_handshake(shm, action_val)

            if i % log_interval == 0:
                latency = (time.perf_counter() - time_start) * 1000
                print(f"\rInference latency: {latency:.2f}ms", end="", flush=True)


def _infer():
    """Coordinate resource initialization, launch background threads, and start loop."""
    state = {
        "is_shutdown": False,
    }

    with (Path(__file__).resolve().parent / "config.json").open() as f:
        config = json.load(f)

    def on_press(key):
        exit_key = Key[config["keys"]["exitKeyName"]]

        if key == exit_key:
            state["is_shutdown"] = True

    listener = Listener(on_press=on_press)
    listener.start()

    shm = init_shm()

    try:
        model = _init_model(config)
        _run_inference_loop(model, shm, state, config)
    finally:
        listener.stop()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    _infer()
