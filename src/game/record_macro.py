"""Contains logic for parsing a .gdr macro, establishing a shared memory bridge between the script and the mod, and records the captured frame and macro gameplay.

Example:
    Start recording the bloodbath macro:
    $ python record.py bloodbath.gdr
"""

import json
import subprocess
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from struct import pack, unpack

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pynput.keyboard import Key, Listener

from game.screen_capture import start_capture_engine

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
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
            cli_path = Path(__file__).parent.parent.parent / "third_party" / "macro_parser"
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
            if macro_fps is not None and round(macro_fps) != _CONFIG["macroFps"]:
                frame_idx = round(round(frame_idx * _CONFIG["macroFps"]) / round(macro_fps))

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
    shm_name = _CONFIG["shmName"]
    try:
        shm = SharedMemory(name=shm_name)
        shm.buf[0:16] = bytes(16)
    except FileNotFoundError:
        shm = SharedMemory(
            name=shm_name,
            create=True,
            size=16,
        )
    return shm


def _active_shm_process(shm: SharedMemory):
    """Run the shared memory loop, processing macro events and updating actions."""
    global _macro_actions_240

    while not _is_shutdown:
        # Extract as integer (i).
        frame_ready_bin = unpack("i", shm.buf[8:12])[0]

        if frame_ready_bin == 1:
            frame_idx = unpack("i", shm.buf[0:4])[0]

            # Slice the 4 actions directly from the pre-populated array
            a = _macro_actions_240[frame_idx : frame_idx + 4]

            # Pack them into the 4 lowest bits
            action_val = int(a[0]) | (int(a[1]) << 1) | (int(a[2]) << 2) | (int(a[3]) << 3)

            # 4:8 - Current action. 12:16 - Python acknowledgement
            shm.buf[4:8] = pack("i", action_val)
            shm.buf[8:12] = pack("i", 0)
            shm.buf[12:16] = pack("i", 1)

        time.sleep(0)  # this bypasses python GIL to allow consistent 120fps recording

    shm.close()
    shm.unlink()


def _run_recording_loop(capture_engine, shm: SharedMemory, frames_buf: np.ndarray, actions_bin_buf: np.ndarray) -> int:
    """Run the main frame and action recording loop, returning the number of frames recorded."""
    buf_max_frames = len(frames_buf)
    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]
    frame_idx = 0

    while not _is_shutdown:
        bgra_frame = capture_engine.queue_full.get()
        frame = bgra_frame[:, :, :3].copy()
        capture_engine.queue_empty.put_nowait(bgra_frame)

        if frame_idx >= buf_max_frames:
            print("Frame buffer exceeded.")
            break

        if _is_recording:
            frames_buf[frame_idx] = frame

            # Read current game tick from shared memory
            current_tick = unpack("i", shm.buf[0:4])[0]

            # Align tick to the start of the 4-tick block (rounding down) to ensure perfect 60Hz frame alignment
            aligned_tick = (current_tick // 4) * 4

            # Get the 4 actions directly via slicing (guaranteed size 4 due to pre-allocated padding)
            actions_bin_buf[frame_idx] = _macro_actions_240[aligned_tick : aligned_tick + 4]
            frame_idx += 1

            if frame_idx % log_interval == 0:
                print(f"\rFrames recorded: {frame_idx}", end="", flush=True)

    return frame_idx


def _record(filepath: Path):
    """Initialize game environment, shared memory, run frame + action pair recording loop."""
    parsed_macro = _parse_macro_file(filepath)
    macro_events = _process_macro(parsed_macro)

    _build_macro_actions_240(macro_events)

    shm = _init_shm()
    shm_thread = threading.Thread(target=_active_shm_process, args=(shm,), daemon=True)
    shm_thread.start()

    listener = Listener(on_press=_on_press)
    listener.start()

    capture_engine = start_capture_engine()

    buf_max_frames = _CONFIG["bufMaxFrames"]
    frame_height_px = _CONFIG["capture"]["frameDims"]["pipelineHeightPx"]
    frame_width_px = _CONFIG["capture"]["frameDims"]["pipelineWidthPx"]

    frames_buf: np.ndarray = np.empty(
        (buf_max_frames, frame_height_px, frame_width_px, 3),
        dtype=np.uint8,
    )
    actions_bin_buf = np.zeros((buf_max_frames, 4), dtype=np.uint8)

    try:
        frame_idx = _run_recording_loop(capture_engine, shm, frames_buf, actions_bin_buf)
    finally:
        listener.stop()
        capture_engine.stop_capture_stream()

    should_save: str = input("\nSave this recording? (Y/n): ")
    if should_save == "n":
        filepath.unlink()
        return

    dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
    dataset_dir: Path = Path(__file__).resolve().parents[2] / dataset_dir_name

    save_path = dataset_dir / f"{filepath.name}-{time.strftime('%m%d%H%M%S')}"
    np.savez_compressed(
        save_path,
        frames=frames_buf[:frame_idx],
        actions_bin=actions_bin_buf[:frame_idx],
    )
    print(f"\nSaved recording to {save_path}")

    filepath.unlink()


if __name__ == "__main__":
    downloads_dir = Path.home() / "Downloads"

    try:
        macro_path = next(downloads_dir.glob("*.gdr"))
    except StopIteration:
        macro_path = next(downloads_dir.glob("*.json"))

    print(f"Using macro: {macro_path}")

    _record(macro_path)
