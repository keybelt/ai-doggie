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

from game.game_env import GameEnv

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
        time.sleep(_CONFIG["recordStartDelaySec"])
        _is_recording = True
        print("Recording started.")
    elif key == Key[exit_key_name]:
        _is_shutdown = True


def _load_macro(filepath: str) -> list[tuple[int, int]]:
    """Unpack .gdr macro files with a modified version of maxnut/gdr-converter's algorithm.

    Returns:
        actions in format (frame, action).
    """

    import msgpack

    macro_events: list[tuple[int, int]] = []

    macro_data = Path(filepath).read_bytes()

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
            result = subprocess.run([str(cli_path), filepath], capture_output=True, text=True)
            parsed_macro = json.loads(result.stdout)
            print("Macro parsed with C++ fallback.")

    macro_fps = parsed_macro.get("framerate")
    print(f"Macro FPS: {macro_fps}.")

    for macro_input in parsed_macro.get("inputs", []):
        frame_idx = macro_input["frame"]
        mouse_btn: int = macro_input["btn"]
        # is_player2 = macro_input.get("2p")
        is_keydown = macro_input["down"]

        if mouse_btn == 1:  # and not is_player2:
            if macro_fps != _CONFIG["macroFps"]:
                frame_idx = round(round(frame_idx * _CONFIG["macroFps"]) / round(macro_fps))

            macro_events.append((frame_idx, 1 if is_keydown else 0))

    macro_events.sort(key=lambda x: x[0])

    print(f"Macro parsed with {len(macro_events)} events.")

    return macro_events


def _shm_bridge(macro_events: list[tuple[int, int]]):
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
            while event_idx < len(macro_events) and frame_idx >= macro_events[event_idx][0]:
                _curr_action_bin = macro_events[event_idx][1]
                event_idx += 1

            # 4:8 - Current action. 12:16 - Python acknowledgement
            shm.buf[4:8] = pack("i", _curr_action_bin)
            shm.buf[8:12] = pack("i", 0)
            shm.buf[12:16] = pack("i", 1)

    shm.close()
    shm.unlink()


def _record(macro_name: str):
    """Initialize game environment, shared memory, run frame + action pair recording loop."""
    buf_max_frames = _CONFIG["bufMaxFrames"]
    frame_height_px = _CONFIG["capture"]["frameDims"]["pipelineHeightPx"]
    frame_width_px = _CONFIG["capture"]["frameDims"]["pipelineWidthPx"]
    log_interval = _CONFIG["logIntervalSec"] * _CONFIG["capture"]["fps"]

    dataset_dir_name = _CONFIG["fileNames"]["datasetDirName"]
    dataset_dir: Path = Path(__file__).resolve().parents[2] / dataset_dir_name

    macro_events: list[tuple[int, int]] = _load_macro(macro_name)

    shm_thread = threading.Thread(target=_shm_bridge, args=(macro_events,), daemon=True)
    shm_thread.start()

    listener = Listener(on_press=_on_press)
    listener.start()

    game_env: GameEnv = GameEnv()

    frames_buf: np.ndarray = np.empty(
        (buf_max_frames, frame_height_px, frame_width_px, 3),
        dtype=np.uint8,
    )
    actions_bin_buf = np.zeros(buf_max_frames, dtype=np.uint8)

    frame_idx = 0
    game_env.clear_frame_queue()

    while not _is_shutdown:
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

    listener.stop()
    game_env.capture_engine.stop_capture_stream()

    should_save: str = input("\nSave this recording? (Y/n): ")
    if should_save == "n":
        return

    save_path = dataset_dir / f"{Path(macro_name).name}-{time.strftime('%m%d%H%M%S')}"
    np.savez_compressed(
        save_path,
        frames=frames_buf[:frame_idx],
        actions_bin=actions_bin_buf[:frame_idx],
    )
    print(f"\nSaved recording to {save_path}")


if __name__ == "__main__":
    downloads_dir = Path.home() / "Downloads"

    try:
        macro_name = str(next(downloads_dir.glob("*.gdr")))
    except StopIteration:
        macro_name = str(next(downloads_dir.glob("*.json")))

    print(f"Using macro: {macro_name}")

    _record(macro_name)
