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


def process_macro(parsed_macro: dict) -> list[dict]:
    """Normalize macro events to raw frame event list.

    Returns:
        Sorted list of raw macro event dicts with frame and down keys.
    """
    raw_events = []
    macro_fps = parsed_macro["framerate"]
    print(f"Macro FPS: {macro_fps}.")
    for macro_input in parsed_macro["inputs"]:
        if macro_input["btn"] == 1:
            raw_events.append(
                {
                    "frame": macro_input["frame"],
                    "down": macro_input["down"],
                }
            )

    raw_events.sort(key=lambda x: x["frame"])
    print(f"Macro processed with {len(raw_events)} events.")
    return raw_events


def build_macro_actions_60(macro_events: list[dict]) -> tuple[np.ndarray, int]:
    """Pre-populate a 60Hz binary action array from raw 240Hz macro events.

    Returns:
        Tuple of (dense_60hz_actions, max_frame_60).
    """
    max_tick_240 = max(e["frame"] for e in macro_events)
    max_frame_60 = (max_tick_240 // 4) + 60
    macro_actions_240 = np.zeros(max_tick_240 + 16, dtype=np.uint8)

    prev_frame = 0
    current_action = 0
    for ev in macro_events:
        frame = ev["frame"]
        down = 1 if ev["down"] else 0
        macro_actions_240[prev_frame:frame] = current_action
        prev_frame = frame
        current_action = down
    macro_actions_240[prev_frame:] = current_action

    num_frames = (len(macro_actions_240) + 3) // 4
    macro_actions_60 = np.zeros(num_frames, dtype=np.uint8)
    for f in range(num_frames):
        macro_actions_60[f] = 1 if np.any(macro_actions_240[f * 4 : (f + 1) * 4] > 0) else 0

    return macro_actions_60, max_frame_60


def run_recording_loop(
    shm: SharedMemory,
    macro_actions_60: np.ndarray,
    max_frame_60: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run 60Hz frame and action recording loop.

    Returns:
        Tuple of (recorded_frames, recorded_actions, is_dead_flag).
    """
    frame_w: int = CONFIG["frame"]["width"]
    frame_h: int = CONFIG["frame"]["height"]
    log_interval: int = max(1, round(CONFIG["logIntervalSec"] * CONFIG["fps"]))

    frames_buf = np.empty((RECORDING_BUFFER_SIZE, frame_h, frame_w, 3), dtype=np.uint8)
    actions_buf = np.zeros(RECORDING_BUFFER_SIZE, dtype=np.uint8)

    frame_idx = 0
    last_frame = -1

    while True:
        current_frame, is_ready = wait_for_next_frame(shm, last_frame)
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

        if current_frame > max_frame_60:
            print("\nMacro finished! Stopping recording...")
            acknowledge_handshake(shm)
            break

        last_frame = current_frame
        raw_frame = get_frame(shm, frame_w, frame_h)
        acknowledge_handshake(shm)

        if frame_idx >= RECORDING_BUFFER_SIZE:
            print("Frame buffer exceeded.")
            break

        action = macro_actions_60[current_frame] if current_frame < len(macro_actions_60) else 0
        frames_buf[frame_idx] = raw_frame
        actions_buf[frame_idx] = action
        frame_idx += 1

        if frame_idx % log_interval == 0:
            print(f"\rFrames recorded: {frame_idx}", end="", flush=True)

    return frames_buf[:frame_idx], actions_bin_buf[:frame_idx], is_dead


def record(filepath: Path):
    """Parse macro, set up shared memory, and record frames into an HDF5 dataset file."""
    parsed_macro = parse_macro_file(filepath)
    macro_events = process_macro(parsed_macro)
    shm = init_shm()

    try:
        frames, actions_bin, is_dead = run_recording_loop(shm, macro_events)
        if is_dead:
            filepath.unlink()
            return

        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        save_path = data_dir / f"{filepath.name}-{time.strftime('%m%d%H%M%S')}.h5"
        with h5py.File(save_path, "w") as f:
            f.create_dataset("frames", data=frames, compression="gzip", compression_opts=4, chunks=(64, 480, 640, 3))
            f.create_dataset("actions_bin", data=actions_bin, compression="lzf", compression_opts=4)
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
