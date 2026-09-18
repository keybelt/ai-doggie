import json
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from shm_utils import (
    acknowledge_handshake,
    close_session,
    get_frame,
    init_shm,
    is_session_active,
    wait_for_next_frame,
)

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

RECORDING_BUFFER_SIZE = CONFIG["data"]["recordingBufferSize"]


def run_recording_loop(
    shm: SharedMemory,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Returns:
        Tuple of (recorded_frames, raw_ttd, is_dead_flag).
    """
    frame_w: int = CONFIG["frame"]["width"]
    frame_h: int = CONFIG["frame"]["height"]
    log_interval: int = max(1, round(CONFIG["logIntervalSec"] * CONFIG["fps"]))

    frames_buf = np.empty((RECORDING_BUFFER_SIZE, frame_h, frame_w, 3), dtype=np.uint8)
    raw_ttd_buf = np.zeros((RECORDING_BUFFER_SIZE, 3), dtype=np.float32)

    frame_idx = 0
    last_frame = -1
    is_dead = False

    try:
        while is_session_active(shm):
            current_frame, is_ready, telem = wait_for_next_frame(shm, last_frame)
            if not is_ready:
                continue

            if frame_idx == 0:
                print("Recording started.")

            if current_frame < last_frame:
                print("\r\033[KDeath detected! Stopping recording...\n")
                is_dead = True
                close_session(shm)
                break

            last_frame = current_frame
            raw_frame = get_frame(shm)
            acknowledge_handshake(shm)

            if frame_idx >= RECORDING_BUFFER_SIZE:
                print("\r\033[KFrame buffer exceeded.")
                close_session(shm)
                break

            ttd_rel = telem["ttd_release"]
            ttd_hold = telem["ttd_hold"]
            ttd_imp = telem["ttd_impulse"]

            frames_buf[frame_idx] = raw_frame
            raw_ttd_buf[frame_idx] = [ttd_rel, ttd_hold, ttd_imp]
            frame_idx += 1

            if frame_idx % log_interval == 0:
                print(
                    f"\r\033[KFrames: {frame_idx} | "
                    f"TTD [R/H/I]: [{ttd_rel:5.1f}, {ttd_hold:5.1f}, {ttd_imp:5.1f}]",
                    end="",
                    flush=True,
                )

        if not is_dead and frame_idx > 0:
            print("\r\033[KLevel completed! Stopping recording...\n")
    except KeyboardInterrupt:
        print("\r\033[KRecording stopped by user (Ctrl+C).\n")
        close_session(shm)

    return (
        frames_buf[:frame_idx],
        raw_ttd_buf[:frame_idx],
        is_dead,
    )


def main(session_name: str):
    shm = init_shm()

    try:
        frames, raw_ttd, is_dead = run_recording_loop(shm)
        if is_dead:
            print("Recording discarded due to player death.")
            return

        data_dir = Path(__file__).resolve().parents[1] / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        save_path = data_dir / f"{session_name}-{time.strftime('%m%d%H%M%S')}.h5"
        with h5py.File(save_path, "w") as f:
            f.create_dataset("frames", data=frames, compression="gzip", compression_opts=4, chunks=(64, 480, 640, 3))
            f.create_dataset("ttd", data=raw_ttd, compression="lzf")
        print(f"\nSaved {len(frames)} frames to {save_path}\n")
    finally:
        close_session(shm)
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    name = sys.argv[1]
    print(f"\nStarting recording session: {name}")
    main(name)
