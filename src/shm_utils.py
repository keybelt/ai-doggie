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
HEADER_SIZE = 40
ACTIONS_BUFFER_SIZE = 8192
FRAME_WIDTH = CONFIG["frame"]["width"]
FRAME_HEIGHT = CONFIG["frame"]["height"]
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT * 3
SHM_SIZE = HEADER_SIZE + ACTIONS_BUFFER_SIZE + FRAME_SIZE  # 929,820 bytes


class GDSharedMemory(SharedMemory):
    def close(self):
        try:
            pack_into("i", self.buf, 0, -1)
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


def acknowledge_handshake(shm: SharedMemory) -> None:
    pack_into("i", shm.buf, 0, 0)


def close_session(shm: SharedMemory) -> None:
    try:
        pack_into("i", shm.buf, 0, -1)
    except Exception:
        pass


def is_session_active(shm: SharedMemory) -> bool:
    return unpack("i", shm.buf[0:4])[0] != -1


def wait_for_rollout_package(shm: SharedMemory) -> bool:
    """Poll for a completed rollout package from C++.

    Returns:
        True if a new rollout package is ready, False otherwise.
    """
    data_ready = unpack("i", shm.buf[0:4])[0]
    if data_ready != 1:
        time.sleep(0.001)
        return False
    return True


def get_rollout_package(shm: SharedMemory) -> dict:
    """Extract the complete atomic rollout package from shared memory.

    Returns:
        dict containing ftd, aux_state [7], actions [L], frame_0 [H, W, 3].
        aux_state: [p1_vx, p1_vy, p1_gravity, p2_vx, p2_vy, p2_gravity, is_holding]
    """
    _, ftd, p1_vx, p1_vy, p1_grav, p2_vx, p2_vy, p2_grav, is_holding, action_len = unpack(
        "i7f2i", shm.buf[0:HEADER_SIZE]
    )

    action_len = max(0, min(action_len, ACTIONS_BUFFER_SIZE))
    actions = np.frombuffer(
        shm.buf[HEADER_SIZE : HEADER_SIZE + action_len],
        dtype=np.int8,
    ).copy()

    frame_offset = HEADER_SIZE + ACTIONS_BUFFER_SIZE
    frame_0 = (
        np.frombuffer(
            shm.buf[frame_offset : frame_offset + FRAME_SIZE],
            dtype=np.uint8,
        )
        .reshape((FRAME_HEIGHT, FRAME_WIDTH, 3))
        .copy()
    )

    aux_state = np.array(
        [
            p1_vx,
            p1_vy,
            p1_grav,
            p2_vx,
            p2_vy,
            p2_grav,
            float(is_holding),
        ],
        dtype=np.float32,
    )

    return {
        "ftd": float(ftd),
        "aux_state": aux_state,
        "actions": actions,
        "frame_0": frame_0,
    }
