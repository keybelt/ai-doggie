import os
import sys
import time
import torch
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
from env import GeometryDashEnv
from model import GDBehavioralCloningModel

CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "gd_model_epoch_15.pt")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

shutdown_flag = False
reset_memory_flag = False


def on_press(key):
    global shutdown_flag, reset_memory_flag
    if key == Key.esc:
        shutdown_flag = True
    elif key == Key.backspace:
        reset_memory_flag = True


listener = keyboard.Listener(on_press=on_press)
listener.start()


def infer():
    global shutdown_flag, reset_memory_flag

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Could not find model weights at {MODEL_PATH}")
        sys.exit(1)

    print("🧠 Loading Impala ResNet + GRU...")
    model = GDBehavioralCloningModel().to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ Model Loaded and set to Eval Mode.")

    print("\n⚠️  FOCUS GEOMETRY DASH NOW... Starting in 3 seconds.")
    time.sleep(3)

    env = GeometryDashEnv()

    hidden_state = None
    current_action = 0
    total_flushed = 0
    step_count = 0
    jumped_in_window = False

    print("\n--- AI IS NOW PLAYING ---")

    latencies = []

    with torch.inference_mode():
        while not shutdown_flag:
            frame, _, flushed_count = env.step(current_action)
            total_flushed += flushed_count

            t_start = time.perf_counter()

            if reset_memory_flag:
                hidden_state = None
                current_action = 0
                reset_memory_flag = False
                env.reset()

            frame_chw = np.transpose(frame, (2, 0, 1))
            frame_tensor = torch.from_numpy(frame_chw).unsqueeze(0).unsqueeze(0)
            frame_tensor = frame_tensor.to(DEVICE, dtype=torch.float32).div(255.0)

            prev_action_tensor = torch.tensor([[current_action]], dtype=torch.long, device=DEVICE)

            logits, hidden_state = model(frame_tensor, prev_action_tensor, hidden_state)

            current_action = torch.argmax(logits, dim=-1).item()
            if current_action == 1:
                jumped_in_window = True

            step_count += 1

            inf_ms = (time.perf_counter() - t_start) * 1000
            latencies.append(inf_ms)
            if len(latencies) > 120:
                latencies.pop(0)

            if step_count % 30 == 0:
                avg_lat = sum(latencies) / len(latencies)
                action_str = "🟩 JUMP" if jumped_in_window else "⬜ IDLE"
                jumped_in_window = False
                true_drops = env.engine.drop_count + total_flushed

                color = "\033[92m" if avg_lat < 6.0 else "\033[93m" if avg_lat < 8.33 else "\033[91m"
                reset_c = "\033[0m"

                sys.stdout.write(
                    f"\r[Live] Action: {action_str} | Latency: {color}{avg_lat:.2f} ms{reset_c} | Frame Drops: {true_drops} ")
                sys.stdout.flush()

    print("\n\n🛑 AI Offline. Shutting down gracefully.")
    env.engine.stop()
    sys.exit(0)


if __name__ == "__main__":
    infer()