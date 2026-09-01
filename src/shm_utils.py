"""Shared memory utilities for communication between Python scripts and Geometry Dash."""

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
HEADER_SIZE = 12  # 3 int32s: frameIdx, frameReadyBin, macroCount
FRAME_SIZE = CONFIG["frame"]["width"] * CONFIG["frame"]["height"] * 3
MAX_MACRO_EVENTS = CONFIG["data"]["recordingBufferSize"]
MACRO_EVENT_SIZE = 8  # 2 int32s: frame, down
SHM_SIZE = HEADER_SIZE + FRAME_SIZE + (MAX_MACRO_EVENTS * MACRO_EVENT_SIZE)
MACRO_OFFSET = HEADER_SIZE + FRAME_SIZE


def init_shm() -> SharedMemory:
    """Create or open POSIX shared memory buffer matching GDMem structure."""
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
    pack_into("i", shm.buf, 8, count)


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


def wait_for_next_frame(shm: SharedMemory, last_tick: int) -> tuple[int, bool]:
    """Checks the shared memory header state to see if a new frame is ready.

    Args:
        shm: SharedMemory object.
        last_tick: The last processed game tick.

    Returns:
        tuple[int, bool]: (current_tick, is_new_frame_ready)
    """
    current_tick, frame_ready = unpack("2i", shm.buf[0:8])

    if frame_ready != 1:
        time.sleep(0)
        return current_tick, False

    if current_tick == last_tick:
        acknowledge_handshake(shm)
        return current_tick, False

    return current_tick, True
