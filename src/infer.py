"""Contains model inference procedure and allows shutdown via pressing the esc key.

Example:
    $ python infer.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float32
from pynput.keyboard import Key, Listener

from game_env import GameEnv
from policy_model import PolicyModel
from type_defs import Frame

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
with _CONFIG_PATH.open("r") as f:
    _CONFIG = json.load(f)

_is_shutdown = False


def _shutdown_on_press(key):
    global _is_shutdown  # noqa: PLW0603

    if key == Key[_CONFIG["keys"]["exitKeyName"]]:
        _is_shutdown = True


_listener = Listener(on_press=_shutdown_on_press)
_listener.start()


def _infer():
    """Load model and weights, track frame drops and latency, and process model output."""
    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    pipeline_fps = _CONFIG["capture"]["fps"]
    device = torch.device(_CONFIG["model"]["deviceName"])

    base_dir = Path(__file__).resolve().parents[1]
    weight_dir: Path = base_dir / _CONFIG["fileNames"]["weightDirName"]
    weight_name = _CONFIG["fileNames"]["weightName"]

    model: PolicyModel = PolicyModel().to(device)
    weights: dict[str, torch.Tensor] = torch.load(
        weight_dir / weight_name,
        weights_only=False,
    )

    model.load_state_dict(weights["model_state"])

    print("Open Geometry Dash within 3 seconds.")
    time.sleep(3)

    env: GameEnv = GameEnv()

    hidden_state: Float32[torch.Tensor, "L N D"] = torch.zeros(
        1,
        1,
        hidden_state_dim,
        device=device,
    )
    curr_action_bin = 0

    i = 0
    with torch.inference_mode():
        while not _is_shutdown:
            i += 1

            frame_HWC: Frame
            frame_HWC, _ = env.step(curr_action_bin)

            time_start: float = time.perf_counter()

            frame_NTHWC: np.uint8 = (
                torch.from_numpy(frame_HWC).unsqueeze(0).unsqueeze(0)
            )
            frame_NTHWC = frame_NTHWC.to(device=device, dtype=torch.float32) / 255

            logits: Float32[torch.Tensor, "N T V"]
            hidden_state: Float32[torch.Tensor, "N L D"]
            logits, hidden_state = model(frame_NTHWC, hidden_state)

            curr_action_bin = torch.argmax(logits, dim=-1).item()

            infer_time: float = (time.perf_counter() - time_start) * 1000

            if i % (120 / pipeline_fps) == 0:
                frame_drops = env.capture_engine.frame_drops
                print(
                    f"\rInference latency: {infer_time} | Frame drops: {frame_drops}",
                    end="",
                    flush=True,
                )

        env.capture_engine.stop_capture_stream()


if __name__ == "__main__":
    _infer()
