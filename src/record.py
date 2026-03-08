import time
import os
import sys
import numpy as np
from pynput import keyboard
from pynput.keyboard import KeyCode, Key
from env import GeometryDashEnv

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATASET_DIR, exist_ok=True)

current_action = 0
trigger_save = False
trigger_reset = False
shutdown_flag = False


def on_press(key):
    global current_action, trigger_save, trigger_reset, shutdown_flag
    if key == KeyCode.from_char('/'):
        current_action = 1
    elif key == Key.enter:
        trigger_save = True
    elif key == Key.backspace:
        trigger_reset = True
    elif key == Key.esc:
        shutdown_flag = True


def on_release(key):
    global current_action
    if key == KeyCode.from_char('/'):
        current_action = 0


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


def save_buffer(frames, actions, count):
    if count == 0:
        print("\n[Save] Buffer is empty. Nothing to save.")
        return

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATASET_DIR, f"run_{timestamp_str}.npz")

    print(f"\n[Save] Formatting {count} frames... This might take a few seconds.")

    np_frames = frames[:count]
    np_actions = actions[:count]

    print(f"[Save] Compressing and writing to disk ({np_frames.nbytes / (1024 ** 3):.2f} GB)...")

    np.savez_compressed(filename, frames=np_frames, actions=np_actions)
    print(f"[Save] Successfully saved Golden Run: {filename}\n")


def record():
    global trigger_save, trigger_reset, shutdown_flag, current_action

    print("⚠️  FOCUS GEOMETRY DASH NOW...")
    time.sleep(3)

    env = GeometryDashEnv()

    BUFFER_CAPACITY = 15000
    frames_buffer = np.empty((BUFFER_CAPACITY, 332, 588, 3), dtype=np.uint8)
    actions_buffer = np.empty(BUFFER_CAPACITY, dtype=np.int8)
    frame_count = 0

    session_saves = 0
    start_time = time.time()

    print("\n--- RECORDING STARTED ---")

    while not shutdown_flag:
        frame, frame_timestamp, _ = env.step(current_action)

        if frame_timestamp == 0.0:
            time.sleep(0.001)
            continue

        frames_buffer[frame_count] = frame
        actions_buffer[frame_count] = current_action
        frame_count += 1

        if trigger_reset:
            frame_count = 0
            trigger_reset = False
            env.reset()
            print("\n[Reset] 🗑️ Buffer cleared! Ready for next attempt.")
            time.sleep(0.5)

        if trigger_save or frame_count >= BUFFER_CAPACITY:
            save_buffer(frames_buffer, actions_buffer, frame_count)
            frame_count = 0
            session_saves += 1
            trigger_save = False
            env.reset()
            print("\n[Save] ✅ Ready for next run.")
            time.sleep(0.5)

        if frame_count % 12 == 0:
            elapsed_sec = time.time() - start_time

            ram_gb = frames_buffer.nbytes / (1024 ** 3)

            c_action = "🟩 JUMP" if current_action == 1 else "⬜ IDLE"

            sys.stdout.write(
                f"\r[Recording] Frames: {frame_count:05d}/{BUFFER_CAPACITY} | "
                f"Time: {elapsed_sec:.1f}s | "
                f"RAM Allocated: {ram_gb:.2f} GB | "
                f"Input: {c_action} | Saves: {session_saves} "
            )
            sys.stdout.flush()

    if frame_count > 0:
        ans = input("\n\nExit triggered. Save the current buffer before quitting? (y/n): ")
        if ans.lower() == 'y':
            save_buffer(frames_buffer, actions_buffer, frame_count)

    print("Recording Session Ended.")
    sys.exit(0)


if __name__ == "__main__":
    record()