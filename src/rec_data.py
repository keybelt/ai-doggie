import sys
from pathlib import Path
from struct import unpack

import h5py

sys.path.append(str(Path(__file__).resolve().parent))

from shm_utils import (
    acknowledge_handshake,
    get_rollout_package,
    init_shm,
    is_session_active,
    wait_for_rollout_package,
)


def update_display(stage_str: str, rollouts: int, first: bool = False):
    if first:
        print(f"[STAGE] {stage_str}\n[DATA]  Rollouts: {rollouts}", end="", flush=True)
    else:
        print(f"\033[A\r\033[K[STAGE] {stage_str}\n\033[K[DATA]  Rollouts: {rollouts}", end="", flush=True)


def main(session_name: str):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    save_path = data_dir / f"{session_name}.h5"
    print(f"\nTarget HDF5 dataset: {save_path}\n")

    shm = init_shm()
    rollout_idx = 0

    try:
        with h5py.File(save_path, "a") as f:
            f.require_group("rollouts")

            update_display("Forward", rollout_idx, first=True)

            while is_session_active(shm):
                data_ready = unpack("i", shm.buf[0:4])[0]
                if data_ready == 2:
                    acknowledge_handshake(shm)
                    update_display("Backward (Golden)", rollout_idx)
                    continue

                if not wait_for_rollout_package(shm):
                    continue

                pkg = get_rollout_package(shm)
                acknowledge_handshake(shm)

                # Save rollout as a self-contained group
                grp = f.create_group(f"rollouts/rollout_{rollout_idx:05d}")
                grp.create_dataset(
                    "frame_0",
                    data=pkg["frame_0"],
                    compression="gzip",
                    compression_opts=4,
                )
                grp.create_dataset("aux_state", data=pkg["aux_state"], dtype="float32")
                grp.create_dataset("actions", data=pkg["actions"], dtype="int8")
                grp.attrs["ftd"] = pkg["ftd"]
                grp.attrs["rollout_idx"] = rollout_idx

                f.flush()

                rollout_idx += 1
                next_stage = "Golden" if rollout_idx % 2 == 0 else "Perturbed"
                update_display(f"Backward ({next_stage})", rollout_idx)

            update_display("Completed", rollout_idx)

    except KeyboardInterrupt:
        print("\n\nRecording stopped by user (Ctrl+C).")
    finally:
        shm.close()
        shm.unlink()
        print(f"\n\nSession finished. Total rollouts saved: {rollout_idx}\n")


if __name__ == "__main__":
    name = sys.argv[1]
    print(f"\nStarting recording session: {name}")
    main(name)
