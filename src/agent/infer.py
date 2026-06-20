"""Contains model inference procedure and allows shutdown via pressing the esc key.

Example:
    $ python infer.py
"""

import json
import sys
import time
from pathlib import Path

import torch
from jaxtyping import Float32
from pynput.keyboard import Key, Listener
from torch import Tensor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent.model import PolicyModel
from game.game_env import GameEnv
from type_defs import Frame

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
    _CONFIG = json.load(f)

_is_inferring = False
_is_shutdown = False


def _on_press(key):
    global _is_shutdown
    global _is_inferring

    if key == Key[_CONFIG["keys"]["exitKeyName"]]:
        _is_shutdown = True
    elif key == Key[_CONFIG["keys"]["recordKeyName"]]:
        _is_inferring = True


def _infer():
    """Load model and weights, track frame drops and latency, and process model output."""
    listener = Listener(on_press=_on_press)
    listener.start()

    hidden_state_dim = _CONFIG["model"]["hiddenDim"]
    device: torch.device = torch.device(_CONFIG["model"]["deviceName"])

    base_dir = Path(__file__).resolve().parents[2]
    checkpoint_dir: Path = base_dir / _CONFIG["fileNames"]["checkpointDirName"]
    checkpoint_name = _CONFIG["fileNames"]["checkpointName"]

    model: PolicyModel = PolicyModel().to(device)
    checkpoint: dict[str, int | float | dict[str, int | Tensor]] = torch.load(
        checkpoint_dir / checkpoint_name,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    model = torch.compile(model)

    env: GameEnv = GameEnv()

    hidden_state: Float32[Tensor, "N L D"] = torch.zeros(
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

            frame_HWC: Frame
            frame_HWC, _ = env.step(curr_action_bin)

            time_start: float = time.perf_counter()

            frame_NTHWC = torch.from_numpy(frame_HWC).unsqueeze(0).unsqueeze(0)
            frame_NTHWC = frame_NTHWC.to(device=device, dtype=torch.float32) / 255

            logits: Float32[Tensor, "N T V"]  # noqa: F722
            hidden_state: Float32[Tensor, "N L D"]  # noqa: F722
            logits, hidden_state = model(frame_NTHWC, hidden_state)

            curr_action_bin = torch.argmax(logits, dim=-1).item()

            infer_time: float = (time.perf_counter() - time_start) * 1000

            if i % log_interval == 0:
                elapsed = time.perf_counter() - now
                drops = env.capture_engine.frame_drops - frame_drop_cache
                print(
                    f"\rInference latency: {infer_time:.2f}ms | Frame drop rate: {drops / elapsed:.2f}/s",
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
