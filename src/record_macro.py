import json
import subprocess
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import h5py
import msgpack
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from shm_utils import acknowledge_handshake, get_frame, init_shm, wait_for_next_frame

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

# Implementation Constants
TARGET_FPS = 60
RECORDING_BUFFER_SIZE = CONFIG["data"]["recordingBufferSize"]


def parse_macro_file(filepath: Path) -> dict:
    """Parse raw .gdr macro file using JSON, msgpack, or C++ CLI parser.

    Returns:
        Unpacked macro dictionary containing inputs and metadata.
    """
    macro_data = filepath.read_bytes()

    try:
        parsed_macro = json.loads(macro_data.decode("utf-8-sig"))
        print("Macro parsed using JSON.")
    except (json.JSONDecodeError, UnicodeDecodeError):
        try:
            parsed_macro = msgpack.unpackb(macro_data, raw=False)
            print("Macro parsed using msgpack.")
        except msgpack.exceptions.ExtraData:
            cli_path = Path(__file__).resolve().parents[1] / "third_party" / "macro_parser"
            result = subprocess.run([str(cli_path), str(filepath)], capture_output=True, text=True)
            parsed_macro = json.loads(result.stdout)
            print("Macro parsed with C++ fallback.")

    return parsed_macro


def process_macro(parsed_macro: dict) -> list[tuple[int, int]]:
    """Normalize macro events to 60Hz frame rate.

    Returns:
        Sorted list of macro event tuples (frame_idx, action_binary).
    """
    macro_events = []
    macro_fps = parsed_macro.get("framerate")
    print(f"Macro FPS: {macro_fps}.")

    for macro_input in parsed_macro.get("inputs", []):
        frame_idx = macro_input["frame"]
        mouse_btn = macro_input["btn"]
        is_keydown = macro_input["down"]

        if mouse_btn == 1:
            if macro_fps is not None and round(macro_fps) != TARGET_FPS:
                frame_idx = round(frame_idx * TARGET_FPS / round(macro_fps))
            macro_events.append((frame_idx, 1 if is_keydown else 0))

    macro_events.sort(key=lambda x: x[0])
    print(f"Macro processed with {len(macro_events)} events.")
    return macro_events


def build_macro_actions_60(macro_events: list[tuple[int, int]]) -> tuple[np.ndarray, int]:
    """Pre-populate a dense 60Hz action array from sparse macro events.

    Returns:
        Tuple of (dense_actions_array, max_frame_index).
    """
    max_frame = max(e[0] for e in macro_events) if macro_events else 0
    macro_actions_60 = np.zeros(max_frame + 1, dtype=np.uint8)

    prev_frame = 0
    current_action = 0
    for frame, action in macro_events:
        macro_actions_60[prev_frame:frame] = current_action
        prev_frame = frame
        current_action = action
    macro_actions_60[prev_frame:] = current_action

    return macro_actions_60, max_frame


def run_recording_loop(
    shm: SharedMemory,
    macro_events: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run 60Hz frame and TTD recording loop.

    Returns:
        Tuple of (recorded_frames, recorded_ttd, is_dead_flag).
    """
    macro_actions_60, max_frame = build_macro_actions_60(macro_events)

    frame_w: int = CONFIG["frame"]["width"]
    frame_h: int = CONFIG["frame"]["height"]
    log_interval: int = CONFIG["logIntervalSec"] * CONFIG["fps"]

    frames_buf = np.empty((RECORDING_BUFFER_SIZE, frame_h, frame_w, 3), dtype=np.uint8)
    ttd_buf = np.zeros((RECORDING_BUFFER_SIZE, 2), dtype=np.int32)

    frame_idx = 0
    last_frame = -1

    while True:
        current_frame, is_ready, ttd_release, ttd_hold = wait_for_next_frame(shm, last_frame)
        if not is_ready:
            continue

        if frame_idx == 0:
            print("Recording started.")

        is_dead = False
        if current_frame < last_frame:
            print("\nDeath detected! Stopping recording...\n")
            is_dead = True
            acknowledge_handshake(shm)
            break

        if current_frame > max_frame:
            print("\nMacro finished! Stopping recording...")
            acknowledge_handshake(shm)
            break

        last_frame = current_frame
        raw_frame = get_frame(shm, frame_w, frame_h)

        action = macro_actions_60[current_frame]
        acknowledge_handshake(shm, int(action))

        if frame_idx >= RECORDING_BUFFER_SIZE:
            print("Frame buffer exceeded.")
            break

        frames_buf[frame_idx] = raw_frame
        ttd_buf[frame_idx] = [ttd_release, ttd_hold]
        frame_idx += 1

        if frame_idx % log_interval == 0:
            print(
                f"\rFrames recorded: {frame_idx} | TTD: [{ttd_release}, {ttd_hold}] | Act: {action}", end="", flush=True
            )

    return frames_buf[:frame_idx], ttd_buf[:frame_idx], is_dead


def record(filepath: Path):
    """Parse macro, set up shared memory, and record frames and TTD into an HDF5 dataset file."""
    parsed_macro = parse_macro_file(filepath)
    macro_events = process_macro(parsed_macro)
    shm = init_shm()

    try:
        frames, ttd, is_dead = run_recording_loop(shm, macro_events)
        if is_dead:
            filepath.unlink()
            return

        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        save_path = data_dir / f"{filepath.name}-{time.strftime('%m%d%H%M%S')}.h5"
        with h5py.File(save_path, "w") as f:
            f.create_dataset("frames", data=frames, compression="gzip", compression_opts=4, chunks=(64, 480, 640, 3))
            f.create_dataset("ttd", data=ttd, compression="lzf")
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
    record(macro_path)
