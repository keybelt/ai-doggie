"""Contains logic for simulating keypress. Handles frame delivery.

Example:
    Get the frame while sending a keydown toggle:
    >>> env = GameEnv()
    >>> frame, _ = env.step(True)
"""

import atexit
import json
import pathlib
import queue

import numpy as np
import Quartz
from AppKit import NSApplication

from capture import _CaptureEngine, start_capture_engine
from type_defs import Frame, FramePackage

with pathlib.Path.open("../config.json") as f:
    _CONFIG_CAPTURE = json.load(f)["capture"]


class _KeyboardController:
    """Processes keyboard actions, and automatically cleans up any stale keypresses."""

    def __init__(self):
        """Initialize key state, keycode variable, and Quartz keyboard event source. Schedule cleanup upon exit."""
        self._is_keydown = False

        atexit.register(self._cleanup)

    def _send_keypress(self, is_keydown):
        # Tell Quartz the events are coming from the Human Interface Device (keyboard/mouse).
        event_src = Quartz.CGEventSourceCreate(
            Quartz.kCGEventSourceStateHIDSystemState,
        )
        e = Quartz.CGEventCreateKeyboardEvent(
            event_src,
            49,  # The space keycode.
            is_keydown,
        )
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, e)

        self._is_keydown = is_keydown

    def _cleanup(self):
        """Release held key and reset the hold state."""
        if self._is_keydown:
            self._send_keypress(is_keydown=False)

    def toggle_key(self, is_keydown):
        """Post key state if key wasn't already in said state."""
        if self._is_keydown != is_keydown:
            self._send_keypress(is_keydown)


class GameEnv:
    """Toggle key inputs, process frames and define a method for clearing the frame queue."""

    def __init__(self):
        """Initialize capture engine, fresh frame buffer, and keyboard controller."""
        frame_height_px = _CONFIG_CAPTURE["frameDims"]["pipelineHeightPx"]
        frame_width_px = _CONFIG_CAPTURE["frameDims"]["pipelineWidthPx"]

        self.capture_engine: _CaptureEngine = start_capture_engine()
        self._keyboard_controller: _KeyboardController = _KeyboardController()

        # 3rd axis represent color channels (dim=3, because dropped alpha). Use uint8 for optimal color representation.
        self._last_fresh_frame: Frame = np.zeros(
            (frame_height_px, frame_width_px, 3),
            dtype=np.uint8,
        )

    def clear_frame_queue(self):
        while self.capture_engine.queue_full.qsize() > 1:
            # FIFO retrieval, dump all the oldest frames.
            frame = self.capture_engine.queue_full.get_nowait()
            self.capture_engine.queue_empty.put(frame)

            self.capture_engine.frame_drops += 1

        print("queue_full cleared.")

    def get_frame(self) -> FramePackage:
        """Return the latest frame, recycle a previous frame if capture_engine doesn't have a fresh one ready.

        Returns:
            The frame in RGB format, and whether the frame is fresh or reused.
        """
        pipeline_fps = _CONFIG_CAPTURE["fps"]

        is_stale = False

        try:
            bgra_frame: Frame = self.capture_engine.queue_full.get(
                timeout=1 / pipeline_fps,
            )
            rgb_frame: Frame = bgra_frame[:, :, 2::-1].copy()

            self.capture_engine.queue_empty.put_nowait(bgra_frame)
            self._last_fresh_frame = rgb_frame

        except queue.Empty:
            rgb_frame: Frame = self._last_fresh_frame
            is_stale = True

        return rgb_frame, is_stale

    def step(self, is_keydown) -> FramePackage:
        """Attempt to toggle the key input.

        Returns:
            The frame in RGB format, and whether the frame is fresh or reused.
        """
        self._keyboard_controller.toggle_key(is_keydown)
        return self.get_frame()


# Initialize macOS GUI infrastructure, required for event posting.
NSApplication.sharedApplication()
