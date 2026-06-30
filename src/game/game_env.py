"""Contains logic for simulating keypress. Handles frame delivery.

Example:
    Get the frame while sending a keydown toggle:
    >>> env = GameEnv()
    >>> frame, _ = env.step(True)
"""

import json
import queue
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from game.screen_capture import _CaptureEngine, start_capture_engine

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
    _CONFIG = json.load(f)

_CONFIG_CAPTURE = _CONFIG["capture"]
CAPTURE_ENGINE_TIMEOUT: int = 10


class GameEnv:
    """Process frames and define a method for clearing the frame queue."""

    def __init__(self):
        """Initialize capture engine, fresh frame buffer."""
        frame_height_px = _CONFIG_CAPTURE["frameDims"]["pipelineHeightPx"]
        frame_width_px = _CONFIG_CAPTURE["frameDims"]["pipelineWidthPx"]

        self.capture_engine: _CaptureEngine = start_capture_engine()

        color_channel_depth = 3
        self._last_fresh_frame = np.zeros(
            (frame_height_px, frame_width_px, color_channel_depth),
            dtype=np.uint8,
        )

        frame = self.capture_engine.queue_full.get(timeout=CAPTURE_ENGINE_TIMEOUT)
        self.capture_engine.queue_empty.put(frame)
        print("Vision connected.")

    def get_frame(self) -> tuple[np.ndarray, bool]:
        """Return the latest frame, recycle a previous frame if capture_engine doesn't have a fresh one ready.

        Returns:
            The frame in RGB format, and whether the frame is fresh or reused.
        """
        is_stale = False
        try:
            # We use a 1.0s timeout so the OS wakes the thread instantly when a frame arrives, 
            # rather than oversleeping on a tiny 8ms timeout which caps FPS to ~60.
            bgra_frame = self.capture_engine.queue_full.get(
                timeout=1.0,
            )
            frame_no_alpha = bgra_frame[:, :, :3].copy()
            self.capture_engine.queue_empty.put_nowait(bgra_frame)
            self._last_fresh_frame = frame_no_alpha

        except queue.Empty:
            frame_no_alpha = self._last_fresh_frame
            is_stale = True

        return frame_no_alpha, is_stale
