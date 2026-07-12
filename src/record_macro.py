"""Contains logic for parsing a .gdr macro, establishing a shared memory bridge between the script and the mod, and records the captured frame and macro gameplay.

Example:
    Start recording the bloodbath macro:
    $ python record.py bloodbath.gdr
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import msgpack
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from shm_utils import init_shm, get_frame, acknowledge_handshake, wait_for_next_frame


def _parse_macro_file(filepath: Path) -> dict:
    """Parse raw .gdr macro file using JSON, msgpack, or C++ fallback.

    Returns:
        The parsed macro dictionary structure.
    """
    macro_data = filepath.read_bytes()

    try:
        # Unpack with utf8 decoding.
        parsed_macro = json.loads(macro_data.decode("utf-8-sig"))
        print("Macro parsed using JSON.")
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Unpack from bytes.
        try:
            parsed_macro = msgpack.unpackb(macro_data, raw=False)
            print("Macro parsed using msgpack.")
        except msgpack.exceptions.ExtraData:
            PROJECT_ROOT = Path(__file__).resolve().parents[1]
            CLI_PATH = PROJECT_ROOT / "third_party" / "macro_parser"
            result = subprocess.run([str(CLI_PATH), str(filepath)], capture_output=True, text=True)
            parsed_macro = json.loads(result.stdout)
            print("Macro parsed with C++ fallback.")

    return parsed_macro


def _process_macro(parsed_macro: dict) -> list[tuple[int, int]]:
    """Interpret the parsed macro structure, normalizing frame rate and sorting events.

    Inspired by maxnut/gdr-converter.

    Returns:
        List of events as (frame_idx, action).
    """
    TARGET_FPS = 240

    macro_events: list[tuple[int, int]] = []
    macro_fps = parsed_macro.get("framerate")
    print(f"Macro FPS: {macro_fps}.")

    for macro_input in parsed_macro.get("inputs", []):
        frame_idx = macro_input["frame"]
        mouse_btn: int = macro_input["btn"]
        is_keydown = macro_input["down"]

        if mouse_btn == 1:
            if macro_fps is not None and round(macro_fps) != TARGET_FPS:
                frame_idx = round(round(frame_idx * TARGET_FPS) / round(macro_fps))

            macro_events.append((frame_idx, 1 if is_keydown else 0))

    macro_events.sort(key=lambda x: x[0])

    print(f"Macro processed with {len(macro_events)} events.")

    return macro_events


def _build_macro_actions_240(macro_events: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    """Pre-populate a dense 240Hz actions array from sparse macro events.

    Uses fast NumPy slice assignments to fill frame intervals with their active state.
    """
    # frame has 4 subticks
    PADDING_BUFFER = 4

    max_frame = max(e[0] for e in macro_events) if macro_events else 0
    # Add a small padding buffer to avoid index errors at the end of recording
    macro_actions_240 = np.zeros(max_frame + PADDING_BUFFER, dtype=np.uint8)

    prev_frame = 0
    current_action = 0
    for frame, action in macro_events:
        macro_actions_240[prev_frame:frame] = current_action
        prev_frame = frame
        current_action = action
    macro_actions_240[prev_frame:] = current_action

    return macro_actions_240, max_frame


def _run_recording_loop(shm, macro_events: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run the main frame and action recording loop, returning the recorded frames, actions, and whether the player died."""
    BUFFER_SIZE = 50000

    macro_actions_240, max_tick = _build_macro_actions_240(macro_events)

    with (Path(__file__).resolve().parent / "config.json").open() as f:
        config = json.load(f)

    frame_w = config["frame"]["width"]
    frame_h = config["frame"]["height"]
    log_interval = config["logIntervalSec"] * config["fps"]

    frames_buf = np.empty((BUFFER_SIZE, frame_h, frame_w, 3), dtype=np.uint8)
    actions_bin_buf = np.zeros((BUFFER_SIZE, 4), dtype=np.uint8)

    frame_idx = 0
    last_tick = -1

    while True:
        # Read header state and wait for next frame
        current_tick, is_ready = wait_for_next_frame(shm, last_tick)
        if not is_ready:
            continue

        if frame_idx == 0:
            print("Recording started.")

        is_dead = False
        if current_tick < last_tick:
            print("\nDeath detected! Stopping recording...\n")
            is_dead = True

            acknowledge_handshake(shm)
            break

        if current_tick > max_tick:
            print("\nMacro finished! Stopping recording...")

            acknowledge_handshake(shm)
            break

        last_tick = current_tick

        # Read frame buffer using shared utility
        raw_frame = get_frame(shm, frame_w, frame_h)

        # Get the 4 actions directly via slicing and pack them
        aligned_tick = (current_tick // 4) * 4
        a = macro_actions_240[aligned_tick : aligned_tick + 4]
        action_val = int(a[0]) | (int(a[1]) << 1) | (int(a[2]) << 2) | (int(a[3]) << 3)

        acknowledge_handshake(shm, action_val)

        if frame_idx >= BUFFER_SIZE:
            print("Frame buffer exceeded.")
            break

        frames_buf[frame_idx] = raw_frame
        actions_bin_buf[frame_idx] = a
        frame_idx += 1

        if frame_idx % log_interval == 0:
            print(f"\rFrames recorded: {frame_idx}", end="", flush=True)

    return frames_buf[:frame_idx], actions_bin_buf[:frame_idx], is_dead


def _record(filepath: Path):
    """Initialize game environment, shared memory, run frame + action pair recording loop."""
    parsed_macro = _parse_macro_file(filepath)
    macro_events = _process_macro(parsed_macro)

    shm = init_shm()

    try:
        frames, actions_bin, is_dead = _run_recording_loop(shm, macro_events)

        if is_dead:
            filepath.unlink()
            return

        PROJECT_ROOT = Path(__file__).resolve().parents[1]
        DATA_DIR = PROJECT_ROOT / "data"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        save_path = DATA_DIR / f"{filepath.name}-{time.strftime('%m%d%H%M%S')}"
        np.savez_compressed(
            save_path,
            frames=frames,
            actions_bin=actions_bin,
        )
        print(f"Saved recording to {save_path}\n")

        filepath.unlink()
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    downloads_dir = Path.home() / "Downloads"

    try:
        macro_path = next(downloads_dir.glob("*.gdr"))
    except StopIteration:
        macro_path = next(downloads_dir.glob("*.json"))

    print(f"\nUsing macro: {macro_path.name}")

    _record(macro_path)
