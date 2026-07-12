"""Shared memory utilities for communication between Python scripts and Geometry Dash."""

from multiprocessing.shared_memory import SharedMemory
from struct import pack, unpack
import time
import numpy as np

SHM_NAME = "GDMem"
SHM_SIZE = 921616  # 16 bytes header + 640 * 480 * 3 bytes frame


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
    shm.buf[0:16] = bytes(16)
    return shm


def get_frame(shm: SharedMemory, width: int, height: int) -> np.ndarray:
    """Read a frame buffer from shared memory starting at offset 16 and reshape it.

    Args:
        shm: The SharedMemory object.
        width: Width of the frame.
        height: Height of the frame.

    Returns:
        np.ndarray: The reshaped frame.
    """
    frame_size = width * height * 3
    return np.frombuffer(
        shm.buf[16 : 16 + frame_size],
        dtype=np.uint8,
    ).reshape((height, width, 3))


def acknowledge_handshake(shm: SharedMemory, action_val: int | None = None) -> None:
    """Acknowledge the C++ handshake, optionally writing an action value.

    If action_val is provided, it updates the action value, sets frameReadyBin = 0,
    and sets actionReadyBin = 1 atomically. Otherwise, it only resets the ready states.
    """
    if action_val is None:
        shm.buf[8:16] = pack("2i", 0, 1)
    else:
        shm.buf[4:16] = pack("3i", action_val, 0, 1)


def wait_for_next_frame(shm: SharedMemory, last_tick: int) -> tuple[int, bool]:
    """Checks the shared memory header state to see if a new frame is ready.

    If a frame is not ready, it sleeps briefly. If the tick matches the last tick,
    it acknowledges the handshake and returns False.

    Args:
        shm: SharedMemory object.
        last_tick: The last processed game tick.

    Returns:
        tuple[int, bool]: (current_tick, is_new_frame_ready)
    """
    current_tick, _, frame_ready, _ = unpack("4i", shm.buf[0:16])

    if frame_ready != 1:
        time.sleep(0)
        return current_tick, False

    if current_tick == last_tick:
        acknowledge_handshake(shm)
        return current_tick, False

    return current_tick, True
