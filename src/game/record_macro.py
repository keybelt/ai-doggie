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

from jaxtyping import UInt8
from pynput.keyboard import Key, Listener

from game.game_env import GameEnv
from type_defs import Frame, ParsedMacro

with (Path(__file__).resolve().parents[1] / "config.json").open() as f:
    _CONFIG = json.load(f)

_curr_action_bin = 0
_is_shutdown = False
_is_recording = False


def _on_press(key):
    """Activate shutdown state upon keypress."""
    global _is_shutdown, _is_recording

    exit_key_name = _CONFIG["keys"]["exitKeyName"]
    record_key_name = _CONFIG["keys"]["recordKeyName"]

    if key == Key[record_key_name]:
        time.sleep(0.5)
        _is_recording = True
        print("Recording started.")
    elif key == Key[exit_key_name]:
        _is_shutdown = True


def _load_macro(filepath: str) -> ParsedMacro:
    """Unpack .gdr macro files using the C++ parser.

    Returns:
        actions in format (frame, action).
    """
    macro_events: ParsedMacro = []

    cli_path = Path(__file__).parent.parent / "gdreplayformat" / "macro_parser"

    result = subprocess.run([str(cli_path), filepath], capture_output=True, text=True)
    parsed_macro = json.loads(result.stdout)

    macro_fps = parsed_macro.get("framerate")
    print(f"Macro FPS: {macro_fps}.")

    for macro_input in parsed_macro.get("inputs", []):
        frame_idx = macro_input["frame"]
        mouse_btn: int = macro_input["btn"]
        is_keydown = macro_input["down"]
        is_p2 = macro_input.get("2p", False)

        if mouse_btn == 1 and not is_p2:
            if macro_fps != _CONFIG["macroFps"]:
                frame_idx = round(frame_idx * _CONFIG["macroFps"] / macro_fps)

            macro_events.append((frame_idx, 1 if is_keydown else 0))

    macro_events.sort(key=lambda x: x[0])

    print(f"Macro parsed with {len(macro_events)} events.")

    return macro_events


def _shm_bridge(macro_events: ParsedMacro):
    """Initialize a shared memory block between this script and the c++ mod."""
    global _curr_action_bin

    shm_name = _CONFIG["shmName"]
    try:
        shm = SharedMemory(name=shm_name)
        shm.buf[0:16] = bytes(16)
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
        frame_ready_bin = unpack("i", shm.buf[8:12])[0]

        if frame_ready_bin == 1:
            frame_idx = unpack("i", shm.buf[0:4])[0]

            # Reset event index on respawn / level restart
            if frame_idx < 5:
                event_idx = 0
                _curr_action_bin = 0

            # Only passes if there are macro events left over, and if our current frame matches or is ahead of the macro event's frame.
            while (
                event_idx < len(macro_events)
                and frame_idx >= macro_events[event_idx][0]
            ):
                _curr_action_bin = macro_events[event_idx][1]
                event_idx += 1

            # 4:8 - Current action. 12:16 - Python acknowledgement
            shm.buf[4:8] = pack("i", _curr_action_bin)
            shm.buf[8:12] = pack("i", 0)
            shm.buf[12:16] = pack("i", 1)
        else:
            time.sleep(0)

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

    frames_buf: UInt8[np.ndarray, "frame_buf_max frame_H frame_W frame_C"] = np.empty(  # noqa: F722
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
        if _is_recording and not is_stale:
            frames_buf[frame_idx] = frame
            actions_bin_buf[frame_idx] = _curr_action_bin
            frame_idx += 1

            if frame_idx % log_interval == 0:
                print(f"\rRecord frames: {frame_idx}", end="", flush=True)

    save_path = dataset_dir / f"{macro_name}-{time.strftime('%m%d%H%M%S')}"
    np.savez_compressed(
        save_path,
        frames=frames_buf[:frame_idx],
        actions_bin=actions_bin_buf[: frame_idx + 1],
    )
    print(f"\nSaved recording to {save_path}")

    game_env.capture_engine.stop_capture_stream()


if __name__ == "__main__":
    macro_name = sys.argv[1]
    _record(macro_name)
