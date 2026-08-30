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

from shm_utils import acknowledge_handshake, get_frame, init_shm, load_macro_to_shm, wait_for_next_frame

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


def process_macro(parsed_macro: dict) -> list[dict]:
    """Extract raw 240Hz jump macro inputs.

    Returns:
        Sorted list of raw macro event dicts with frame and down.
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


def run_recording_loop(
    shm: SharedMemory,
    max_frame_60: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Run 60Hz frame and TTD recording loop.

    Returns:
        Tuple of (recorded_frames, recorded_ttd, is_dead_flag).
    """
    frame_w: int = CONFIG["frame"]["width"]
    frame_h: int = CONFIG["frame"]["height"]
    log_interval: int = max(1, round(CONFIG["logIntervalSec"] * CONFIG["fps"]))

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

        frames_buf[frame_idx] = raw_frame
        ttd_buf[frame_idx] = [ttd_release, ttd_hold]
        frame_idx += 1

        if frame_idx % log_interval == 0:
            print(
                f"\r\033[KFrames recorded: {frame_idx} | TTD: [{ttd_release:>3}, {ttd_hold:>3}]",
                end="",
                flush=True,
            )

    return frames_buf[:frame_idx], ttd_buf[:frame_idx], is_dead


def record(filepath: Path):
    """Parse macro, set up shared memory, and record frames and TTD into an HDF5 dataset file."""
    parsed_macro = parse_macro_file(filepath)
    macro_events = process_macro(parsed_macro)
    shm = init_shm()
    load_macro_to_shm(shm, macro_events)

    max_frame_60 = (macro_events[-1]["frame"] // 4) + 60

    try:
        frames, ttd, is_dead = run_recording_loop(shm, max_frame_60)
        min_seq_len = CONFIG["training"]["seqLen"]
        if len(frames) < min_seq_len:
            print(f"Recording too short ({len(frames)} frames < {min_seq_len}), discarding.\n")
            filepath.unlink()
            return

        data_dir = Path(__file__).resolve().parents[1] / "data"

        save_path = data_dir / f"{filepath.name}-{time.strftime('%m%d%H%M%S')}.h5"
        with h5py.File(save_path, "w") as f:
            f.create_dataset("frames", data=frames, compression="gzip", compression_opts=4, chunks=(64, 480, 640, 3))
            f.create_dataset("ttd", data=ttd, compression="lzf")
        print(f"Saved {'partial' if is_dead else 'complete'} recording ({len(frames)} frames) to {save_path}\n")
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
