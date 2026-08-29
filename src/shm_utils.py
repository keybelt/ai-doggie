"""Shared memory utilities for communication between Python scripts and Geometry Dash."""

import time
from multiprocessing.shared_memory import SharedMemory
from struct import pack, unpack

import numpy as np

SHM_NAME = "GDMem"
SHM_SIZE = 921624  # 24 bytes header (6 int32s) + 640 * 480 * 3 bytes frame


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
    shm.buf[0:24] = bytes(24)
    return shm


def get_frame(shm: SharedMemory, width: int, height: int) -> np.ndarray:
    """Read a frame buffer from shared memory starting at offset 24 and reshape it.

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
            shm.buf[24 : 24 + frame_size],
            dtype=np.uint8,
        )
        .reshape((height, width, 3))
        .copy()
    )


def acknowledge_handshake(shm: SharedMemory, action_val: int | None = None) -> None:
    """Acknowledge the C++ handshake, optionally writing an action value.

    If action_val is provided, it updates the action value, sets frameReadyBin = 0,
    and sets actionReadyBin = 1 atomically. Otherwise, it only resets the ready states.
    """
    if action_val is None:
        shm.buf[8:16] = pack("2i", 0, 1)
    else:
        shm.buf[4:16] = pack("3i", action_val, 0, 1)


def get_ttd(shm: SharedMemory) -> tuple[int, int]:
    """Read ttdRelease and ttdHold from shared memory header.

    Returns:
        tuple[int, int]: (ttd_release, ttd_hold)
    """
    ttd_release, ttd_hold = unpack("2i", shm.buf[16:24])
    return ttd_release, ttd_hold


def wait_for_next_frame(shm: SharedMemory, last_tick: int) -> tuple[int, bool, int, int]:
    """Checks the shared memory header state to see if a new frame is ready.

    If a frame is not ready, it sleeps briefly. If the tick matches the last tick,
    it acknowledges the handshake and returns False.

    Returns:
        tuple[int, bool, int, int]: (current_tick, is_new_frame_ready, ttd_release, ttd_hold)
    """
    current_tick, _, frame_ready, _, ttd_release, ttd_hold = unpack("6i", shm.buf[0:24])

    if frame_ready != 1:
        time.sleep(0)
        return current_tick, False, ttd_release, ttd_hold

    if current_tick == last_tick:
        acknowledge_handshake(shm)
        return current_tick, False, ttd_release, ttd_hold

    return current_tick, True, ttd_release, ttd_hold
