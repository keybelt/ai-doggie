import json
import time
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from struct import pack_into, unpack

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with CONFIG_PATH.open() as f:
    CONFIG = json.load(f)

SHM_NAME = "GDMem"
HEADER_SIZE = 20  # 5 int32s: frameIdx, frameReadyBin, ttdRelease, ttdHold, macroCount
FRAME_SIZE = CONFIG["frame"]["width"] * CONFIG["frame"]["height"] * 3
MAX_MACRO_EVENTS = CONFIG["data"]["recordingBufferSize"]
MACRO_EVENT_SIZE = 8  # 2 int32s: frame, down
SHM_SIZE = HEADER_SIZE + FRAME_SIZE + (MAX_MACRO_EVENTS * MACRO_EVENT_SIZE)
MACRO_OFFSET = HEADER_SIZE + FRAME_SIZE


def init_shm() -> SharedMemory:
    """Initialize a shared memory block between the Python script and the C++ mod."""
    try:
        shm = SharedMemory(name=SHM_NAME)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

    shm = SharedMemory(
        name=SHM_NAME,
        create=True,
        size=SHM_SIZE,
    )
    shm.buf[0:HEADER_SIZE] = bytes(HEADER_SIZE)
    return shm


def load_macro_to_shm(shm: SharedMemory, macro_events: list[dict]) -> None:
    """Write raw 240Hz macro events (frame, down) into the shared memory macro buffer."""
    count = min(len(macro_events), MAX_MACRO_EVENTS)
    for i in range(count):
        ev = macro_events[i]
        offset = MACRO_OFFSET + (i * MACRO_EVENT_SIZE)
        pack_into(
            "2i",
            shm.buf,
            offset,
            int(ev["frame"]),
            1 if ev["down"] else 0,
        )
    # Set macroCount at offset 16
    pack_into("i", shm.buf, 16, count)


def get_frame(shm: SharedMemory, width: int, height: int) -> np.ndarray:
    """Read a frame buffer from shared memory starting at offset HEADER_SIZE and reshape it.

    Args:
        shm: The SharedMemory object.
        width: Width of the frame.
        height: Height of the frame.

    Returns:
        np.ndarray: The reshaped frame copy.
    """
    frame_size = width * height * 3
    return (
        np.frombuffer(
            shm.buf[HEADER_SIZE : HEADER_SIZE + frame_size],
            dtype=np.uint8,
        )
        .reshape((height, width, 3))
        .copy()
    )


def acknowledge_handshake(shm: SharedMemory) -> None:
    """Reset frameReadyBin to 0 to signal C++ that the frame was consumed."""
    pack_into("i", shm.buf, 4, 0)


def get_ttd(shm: SharedMemory) -> tuple[int, int]:
    """Read ttdRelease and ttdHold from shared memory header.

    Returns:
        tuple[int, int]: (ttd_release, ttd_hold)
    """
    ttd_release, ttd_hold = unpack("2i", shm.buf[8:16])
    return ttd_release, ttd_hold


def wait_for_next_frame(shm: SharedMemory, last_tick: int) -> tuple[int, bool, int, int]:
    """Checks the shared memory header state to see if a new frame is ready.

    If a frame is not ready, it sleeps briefly. If the tick matches the last tick,
    it acknowledges the handshake and returns False.

    Returns:
        tuple[int, bool, int, int]: (current_tick, is_new_frame_ready, ttd_release, ttd_hold)
    """
    current_tick, frame_ready, ttd_release, ttd_hold, _ = unpack("5i", shm.buf[0:HEADER_SIZE])

    if frame_ready != 1:
        time.sleep(0)
        return current_tick, False, ttd_release, ttd_hold

    if current_tick == last_tick:
        acknowledge_handshake(shm)
        return current_tick, False, ttd_release, ttd_hold

    return current_tick, True, ttd_release, ttd_hold
