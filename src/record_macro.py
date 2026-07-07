"""Contains logic for parsing a .gdr macro, establishing a shared memory bridge between the script and the mod, and records the captured frame and macro gameplay.

Example:
    Start recording the bloodbath macro:
    $ python record.py bloodbath.gdr
"""

import json
import subprocess
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from struct import pack, unpack

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from pynput.keyboard import Key, Listener

with (Path(__file__).resolve().parent / "config.json").open() as f:
    _CONFIG = json.load(f)


_macro_actions_240 = None
_is_shutdown = False
_is_recording = False


def _on_press(key):
    """Activate shutdown state upon keypress."""
    global _is_shutdown, _is_recording

    exit_key_name = _CONFIG["keys"]["exitKeyName"]
    record_key_name = _CONFIG["keys"]["recordKeyName"]

    if key == Key[record_key_name]:
        time.sleep(_CONFIG["recordStartDelaySec"])
        _is_recording = True
        print("Recording started.")
    elif key == Key[exit_key_name]:
        _is_shutdown = True
        print("\nSaving...")


def _parse_macro_file(filepath: Path) -> dict:
    """Parse raw .gdr macro file using JSON, msgpack, or C++ fallback.

    Returns:
        The parsed macro dictionary structure.
    """
    import msgpack

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
            cli_path = Path(__file__).parent.parent / "third_party" / "macro_parser"
            result = subprocess.run([str(cli_path), str(filepath)], capture_output=True, text=True)
            parsed_macro = json.loads(result.stdout)
            print("Macro parsed with C++ fallback.")

    return parsed_macro


def _process_macro(parsed_macro: dict) -> list[tuple[int, int]]:
    """Interpret the parsed macro structure, normalizing frame rate and sorting events.

    Inspired by maxnut/gdr-converter.

    Returns:
        List of events as (frame_idx, action).
    """
    macro_events: list[tuple[int, int]] = []
    macro_fps = parsed_macro.get("framerate")
    print(f"Macro FPS: {macro_fps}.")

    for macro_input in parsed_macro.get("inputs", []):
        frame_idx = macro_input["frame"]
        mouse_btn: int = macro_input["btn"]
        # is_player2 = macro_input.get("2p")
        is_keydown = macro_input["down"]

        if mouse_btn == 1:  # and not is_player2:
            if macro_fps is not None and round(macro_fps) != 240:
                frame_idx = round(round(frame_idx * 240) / round(macro_fps))

            macro_events.append((frame_idx, 1 if is_keydown else 0))

    macro_events.sort(key=lambda x: x[0])

    print(f"Macro processed with {len(macro_events)} events.")

    return macro_events


def _build_macro_actions_240(macro_events: list[tuple[int, int]]):
    """Pre-populate a dense 240Hz actions array from sparse macro events.

    Uses fast NumPy slice assignments to fill frame intervals with their active state.
    """
    global _macro_actions_240

    max_frame = max(e[0] for e in macro_events) if macro_events else 0
    # Add a generous padding buffer (100,000 ticks) to avoid index errors at the end of recording
    _macro_actions_240 = np.zeros(max_frame + 100000, dtype=np.uint8)

    prev_frame = 0
    current_action = 0
    for frame, action in macro_events:
        _macro_actions_240[prev_frame:frame] = current_action
        prev_frame = frame
        current_action = action
    _macro_actions_240[prev_frame:] = current_action


def _init_shm() -> SharedMemory:
    """Initialize a shared memory block between this script and the c++ mod."""
    shm_name = "GDMem"
    try:
        shm = SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

    shm = SharedMemory(
        name=shm_name,
        create=True,
        size=921616,
    )
    shm.buf[0:16] = bytes(16)
    return shm


def _run_recording_loop(shm: SharedMemory, frames_buf: np.ndarray, actions_bin_buf: np.ndarray) -> (int, bool):
    """Run the main frame and action recording loop, returning the number of frames recorded and whether to save."""
    buf_max_frames = len(frames_buf)
    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["fps"]
    frame_idx = 0
    last_tick = -1

    while not _is_shutdown:
        # Wait for C++ to signal frameReadyBin == 1
        frame_ready = unpack("i", shm.buf[8:12])[0]
        if frame_ready != 1:
            time.sleep(0)
            continue

        # Read current game tick from shared memory
        current_tick = unpack("i", shm.buf[0:4])[0]
        if current_tick == last_tick:
            shm.buf[12:16] = pack("i", 1)
            shm.buf[8:12] = pack("i", 0)
            continue

        is_dead = False
        if current_tick < last_tick:
            print("\nDeath detected! Stopping recording...")
            is_dead = True

            # Handshake acknowledgement: set actionReadyBin = 1, frameReadyBin = 0
            shm.buf[12:16] = pack("i", 1)
            shm.buf[8:12] = pack("i", 0)
            break

        last_tick = current_tick

        frame_w = _CONFIG["frame"]["width"]
        frame_h = _CONFIG["frame"]["height"]

        # Read frame buffer (fixed 640x480 resolution starting at offset 16)
        frame_size = frame_w * frame_h * 3
        raw_frame = np.frombuffer(shm.buf[16 : 16 + frame_size], dtype=np.uint8).reshape((frame_h, frame_w, 3))

        # Get the 4 actions directly via slicing and pack them
        aligned_tick = (current_tick // 4) * 4
        a = _macro_actions_240[aligned_tick : aligned_tick + 4]
        action_val = int(a[0]) | (int(a[1]) << 1) | (int(a[2]) << 2) | (int(a[3]) << 3)

        # Write current action to shared memory
        shm.buf[4:8] = pack("i", action_val)

        if frame_idx >= buf_max_frames:
            print("Frame buffer exceeded.")
            shm.buf[12:16] = pack("i", 1)
            shm.buf[8:12] = pack("i", 0)
            break

        if _is_recording:
            frames_buf[frame_idx] = raw_frame
            actions_bin_buf[frame_idx] = a
            frame_idx += 1

            if frame_idx % log_interval == 0:
                print(f"\rFrames recorded: {frame_idx}", end="", flush=True)

        # Handshake acknowledgement: set actionReadyBin = 1, frameReadyBin = 0
        shm.buf[12:16] = pack("i", 1)
        shm.buf[8:12] = pack("i", 0)

    return frame_idx, is_dead


def _record(filepath: Path):
    """Initialize game environment, shared memory, run frame + action pair recording loop."""
    parsed_macro = _parse_macro_file(filepath)
    macro_events = _process_macro(parsed_macro)

    _build_macro_actions_240(macro_events)

    shm = _init_shm()

    try:
        listener = Listener(on_press=_on_press)
        listener.start()

        buffer_size = 50000
        frame_h = _CONFIG["frame"]["height"]
        frame_w = _CONFIG["frame"]["width"]

        frames_buf: np.ndarray = np.empty((buffer_size, frame_h, frame_w, 3), dtype=np.uint8)
        actions_bin_buf = np.zeros((buffer_size, 4), dtype=np.uint8)

        try:
            frame_idx, is_dead = _run_recording_loop(shm, frames_buf, actions_bin_buf)
        finally:
            listener.stop()

        if is_dead:
            filepath.unlink()
            return

        dataset_dir_path: Path = Path(__file__).resolve().parents[1] / "data"
        dataset_dir_path.mkdir(parents=True, exist_ok=True)

        save_path = dataset_dir_path / f"{filepath.name}-{time.strftime('%m%d%H%M%S')}"
        np.savez_compressed(
            save_path,
            frames=frames_buf[:frame_idx],
            actions_bin=actions_bin_buf[:frame_idx],
        )
        print(f"\nSaved recording to {save_path}")

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

    print(f"Using macro: {macro_path}")

    _record(macro_path)
