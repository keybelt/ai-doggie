"""Contains logic for parsing a .gdr macro, establishing a shared memory bridge between the script and the mod, and records the captured frame and macro gameplay.

Example:
    Start recording the bloodbath macro:
    $ python record.py bloodbath.gdr
"""

import json
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from struct import pack, unpack

import msgpack
import numpy as np
from jaxtyping import UInt8
from pynput.keyboard import Key, Listener

from config import CONFIG as _CONFIG
from game_env import GameEnv
from type_defs import Frame, ParsedMacro

_curr_action_bin = 0
_is_shutdown = False
_is_recording = False


def _on_press(key):
    """Activate shutdown state upon keypress."""
    global _is_shutdown, _is_recording  # noqa: PLW0603

    exit_key_name = _CONFIG["keys"]["exitKeyName"]
    record_key_name = _CONFIG["keys"]["recordKeyName"]

    if key == Key[record_key_name]:
        time.sleep(0.5)
        _is_recording = True
        print("Recording started.")
    elif key == Key[exit_key_name]:
        _is_shutdown = True


def _load_macro(filepath: str) -> ParsedMacro:
    """Unpack .gdr macro files with a modified version of maxnut/gdr-converter's algorithm.

    Returns:
        actions in format (frame, action).
    """
    macro_events: ParsedMacro = []

    macro_data = Path(filepath).read_bytes()

    try:
        # Unpack with utf8 decoding.
        parsed_macro = json.loads(macro_data.decode("utf-8-sig"))

        print("Macro parsed with JSON.")
    except json.JSONDecodeError:
        # Unpack from bytes.
        parsed_macro = msgpack.unpackb(macro_data, raw=False)

        print("Macro parsed with MessagePack.")

    for macro_input in parsed_macro.get("inputs", []):
        frame_idx = macro_input["frame"]
        mouse_btn: int = macro_input["btn"]
        is_player2 = macro_input["2p"]
        is_keydown = macro_input["down"]

        if mouse_btn != 1 or is_player2:
            continue

        macro_events.append((frame_idx, 1 if is_keydown else 0))

    macro_events.sort(key=lambda x: x[0])

    print(f"Macro parsed with {len(macro_events)} events.")

    return macro_events


def _shm_bridge(macro_events: ParsedMacro):
    """Initialize a shared memory block between this script and the c++ mod."""
    global _curr_action_bin  # noqa: PLW0603

    shm_name = _CONFIG["shmName"]
    try:
        shm = SharedMemory(name=shm_name)
    except FileNotFoundError:
        shm: SharedMemory = SharedMemory(
            name=shm_name,
            create=True,
            size=16,
        )

    event_idx = 0
    _curr_action_bin = 0

    while not _is_shutdown:
        # Extract as integer (i).
        action_ready_bin = unpack("i", shm.buf[8:12])[0]

        if action_ready_bin == 1:
            frame_idx = unpack("i", shm.buf[0:4])[0]

            # Only passes if there are macro events left over, and if our current frame matches or is ahead of the macro event's frame.
            while (
                event_idx < len(macro_events)
                and frame_idx >= macro_events[event_idx][0]
            ):
                _curr_action_bin = macro_events[event_idx][1]
                event_idx += 1

            # 4:8 - Current action. 12:16 - Python acknowledgement
            shm.buf[4:8] = pack("i", _curr_action_bin)
            shm.buf[12:16] = pack("i", 1)
            shm.buf[8:12] = pack("i", 0)
        else:
            time.sleep(0.001)

    shm.close()
    shm.unlink()


def _record(macro_name: str):
    """Initialize game environment, shared memory, run frame + action pair recording loop."""
    buf_max_frames = _CONFIG["bufMaxFrames"]
    frame_height_px = _CONFIG["capture"]["frameDims"]["pipelineHeightPx"]
    frame_width_px = _CONFIG["capture"]["frameDims"]["pipelineWidthPx"]
    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]

    dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
    dataset_dir: Path = Path(__file__).resolve().parents[1] / dataset_dir_name

    macro_events: ParsedMacro = _load_macro(macro_name)

    shm_thread = threading.Thread(target=_shm_bridge, args=(macro_events,), daemon=True)
    shm_thread.start()

    game_env: GameEnv = GameEnv()

    listener = Listener(on_press=_on_press)
    listener.start()

    frames_buf: UInt8[np.ndarray, "frame_buf_max frame_H frame_W frame_C"] = np.empty(
        (buf_max_frames, frame_height_px, frame_width_px, 3),
        dtype=np.uint8,
    )
    # Store an extra for next action prediction.
    actions_bin_buf = np.zeros(buf_max_frames + 1, dtype=np.uint8)

    frame_idx = 0
    game_env.clear_frame_queue()

    while not _is_shutdown:
        frame: Frame
        frame, is_stale = game_env.get_frame()

        if frame_idx >= buf_max_frames:
            print("Frame buffer exceeded.")
            break

        game_env.clear_frame_queue()
        if _is_recording:
            if is_stale:
                continue

            frames_buf[frame_idx] = frame
            actions_bin_buf[frame_idx] = _curr_action_bin
            frame_idx += 1

            if frame_idx % log_interval == 0:
                print(f"\rRecord frames: {frame_idx}", end="", flush=True)

    save_path = dataset_dir / f"{macro_name}-{time.strftime('%m%d%H%M')}"
    np.savez_compressed(
        save_path,
        frames=frames_buf[:frame_idx],
        actions_bin=actions_bin_buf[: frame_idx + 1],
    )
    print(f"Saved recording to {save_path}")

    game_env.capture_engine.stop_capture_stream()


if __name__ == "__main__":
    macro_name = sys.argv[1]
    _record(macro_name)
