"""Contains model inference procedure and allows shutdown via pressing the esc key.

Example:
    $ python infer.py
"""

import time
from pathlib import Path

import torch
from jaxtyping import Float32
from pynput.keyboard import Key, Listener
from torch import Tensor

from config import CONFIG as _CONFIG
from game_env import GameEnv
from policy_model import PolicyModel
from type_defs import Frame

_is_shutdown = False


def _shutdown_on_press(key):
    global _is_shutdown  # noqa: PLW0603

    if key == Key[_CONFIG["keys"]["exitKeyName"]]:
        _is_shutdown = True


def _infer():
    """Load model and weights, track frame drops and latency, and process model output."""
    listener = Listener(on_press=_shutdown_on_press)
    listener.start()

    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    pipeline_fps = _CONFIG["capture"]["fps"]
    device: torch.device = torch.device(_CONFIG["model"]["deviceName"])

    base_dir = Path(__file__).resolve().parents[1]
    checkpoint_dir: Path = base_dir / _CONFIG["fileNames"]["checkpointDirName"]
    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]

    model: PolicyModel = PolicyModel().to(device)
    checkpoint: dict[str, int | float | dict[str, int | Tensor]] = torch.load(
        checkpoint_dir / checkpoint_name,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state"])

    print("Open Geometry Dash within 3 seconds.")
    time.sleep(3)

    env: GameEnv = GameEnv()

    hidden_state: Float32[Tensor, "N L D"] = torch.zeros(
        1,
        1,
        hidden_state_dim,
        device=device,
    )
    curr_action_bin = 0

    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]
    i = 0
    with torch.inference_mode():
        while not _is_shutdown:
            i += 1

            frame_HWC: Frame
            frame_HWC, _ = env.step(curr_action_bin)

            time_start: float = time.perf_counter()

            frame_NTHWC = torch.from_numpy(frame_HWC).unsqueeze(0).unsqueeze(0)
            frame_NTHWC = frame_NTHWC.to(device=device, dtype=torch.float32) / 255

            logits: Float32[Tensor, "N T V"]
            hidden_state: Float32[Tensor, "N L D"]
            logits, hidden_state = model(frame_NTHWC, hidden_state)

            curr_action_bin = torch.argmax(logits, dim=-1).item()

            infer_time: float = (time.perf_counter() - time_start) * 1000

            if i % log_interval == 0:
                frame_drops = env.capture_engine.frame_drops
                print(
                    f"\rInference latency: {infer_time} | Frame drops: {frame_drops}",
                    end="",
                    flush=True,
                )

    env.capture_engine.stop_capture_stream()


if __name__ == "__main__":
    _infer()
