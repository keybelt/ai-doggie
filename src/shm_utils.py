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
HEADER_SIZE = 20
FRAME_WIDTH = CONFIG["frame"]["width"]
FRAME_HEIGHT = CONFIG["frame"]["height"]
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3
SHM_SIZE = HEADER_SIZE + FRAME_SIZE


class GDSharedMemory(SharedMemory):
    def close(self):
        try:
            pack_into("i", self.buf, 4, -1)
        except Exception:
            pass
        super().close()


def init_shm() -> GDSharedMemory:
    try:
        shm = SharedMemory(name=SHM_NAME)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass

    shm = GDSharedMemory(
        name=SHM_NAME,
        create=True,
        size=SHM_SIZE,
    )
    shm.buf[0:HEADER_SIZE] = bytes(HEADER_SIZE)
    return shm


def get_frame(shm: SharedMemory) -> np.ndarray:
    """
    Returns:
        ndarray representation of frame in height,width,alpha
    """
    return (
        np.frombuffer(
            shm.buf[HEADER_SIZE : HEADER_SIZE + FRAME_SIZE],
            dtype=np.uint8,
        )
        .reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        .copy()
    )


def acknowledge_handshake(shm: SharedMemory) -> None:
    pack_into("i", shm.buf, 4, 0)


def close_session(shm: SharedMemory) -> None:
    try:
        pack_into("i", shm.buf, 4, -1)
    except Exception:
        pass


def get_telemetry(shm: SharedMemory) -> dict[str, float]:
    """
    Returns:
        dict of ttd_release, ttd_hold, ttd_impulse (raw 60Hz frames).
    """
    ttd_rel, ttd_hold, ttd_imp = unpack("3f", shm.buf[8:20])
    return {
        "ttd_release": float(ttd_rel),
        "ttd_hold": float(ttd_hold),
        "ttd_impulse": float(ttd_imp),
    }


def is_session_active(shm: SharedMemory) -> bool:
    return unpack("i", shm.buf[4:8])[0] != -1


def wait_for_next_frame(shm: SharedMemory, last_tick: int) -> tuple[int, bool, dict[str, float]]:
    """
    Returns:
        current_tick, is_new_frame_ready, telemetry_dict
    """
    current_tick, frame_ready = unpack("2i", shm.buf[0:8])
    telemetry = get_telemetry(shm)

    if frame_ready != 1:
        time.sleep(0)
        return current_tick, False, telemetry

    if current_tick == last_tick:
        acknowledge_handshake(shm)
        return current_tick, False, telemetry

    return current_tick, True, telemetry
