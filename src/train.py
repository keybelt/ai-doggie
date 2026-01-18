import torch
import time
import sys
import os
import signal
from collections import deque
from env import GeometryDashEnv
from model import GDPolicy, PPOAgent

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RUN_NAME = "Pretrain"
MAX_TIMESTEPS = 20_000_000
ROLLOUT_STEPS = 1024
SAVE_INTERVAL = 100
LOAD_CHECKPOINT = None


# --- DASHBOARD ---
class ControlRoom:
    def __init__(self, loaded_from, start_step=0):
        self.total_frames = start_step  # [FIX] Initialize with saved steps
        self.update_count = 0
        self.loaded_from = loaded_from if loaded_from else "Fresh Start"

        self.latencies = deque(maxlen=20)
        self.confidences = deque(maxlen=20)
        self.session_deaths = 0

    def log_step(self, inference_ms, confidence):
        self.latencies.append(inference_ms)
        self.confidences.append(confidence)

    def render(self):
        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_conf = sum(self.confidences) / len(self.confidences) * 100 if self.confidences else 0

        # Calculate progress relative to the NEXT save
        progress = self.update_count % SAVE_INTERVAL
        bar_len = 10
        filled = int((progress / SAVE_INTERVAL) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lat_color = "\033[92m" if avg_lat < 8.0 else "\033[91m"
        reset = "\033[0m"

        sys.stdout.write("\033[H\033[J")
        dashboard = f"""
============================================================
   AI Zoink | {RUN_NAME} | LIVE
============================================================
 [SOURCE]
  > Loaded From:       {self.loaded_from}
  > Starting Step:     {self.total_frames:,}

 [PERFORMANCE]
  > Inference Latency: {lat_color}{avg_lat:.2f} ms{reset}
  > AI Confidence:     {avg_conf:.1f}%

 [STATS]
  > Deaths (Session):  {self.session_deaths}

 [TRAINING PROGRESS]
  > Global Steps:      {self.total_frames:,}
  > Weight Updates:    {self.update_count} (Session)
  > Next Checkpoint:   [{bar}] {progress}/{SAVE_INTERVAL}

============================================================
"""
        sys.stdout.write(dashboard)
        sys.stdout.flush()


# --- HANDLERS ---
shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)


def save_checkpoint(model, agent, step_count, filename):
    """Saves Model, Optimizer, and Step Count together"""
    print(f"\n💾 Saving Bundle: {filename}")
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'global_step': step_count
    }
    torch.save(payload, filename)


def train():
    global shutdown_flag
    print("⚠️  FOCUS GEOMETRY DASH NOW...")
    time.sleep(5)

    env = GeometryDashEnv()
    device = torch.device("mps")
    model = GDPolicy().to(device)
    agent = PPOAgent(model)  # Initialize agent early to load optimizer

    # --- SMART LOAD LOGIC ---
    start_step = 0
    loaded_name = None

    if LOAD_CHECKPOINT:
        # Clean path logic
        clean_name = os.path.basename(LOAD_CHECKPOINT)
        path = os.path.join(CHECKPOINT_DIR, clean_name)

        if os.path.exists(path):
            print(f"Loading: {path}")
            checkpoint = torch.load(path, map_location=device)

            # CHECK: Is this a new bundle or old weight file?
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                print("✅ Detected Bundle (Model + Stats)")
                model.load_state_dict(checkpoint['model_state'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_step = checkpoint['global_step']
            else:
                print("⚠️  Detected Legacy Weights (No Stats)")
                model.load_state_dict(checkpoint)
                start_step = 0  # Cannot recover steps from old files

            loaded_name = clean_name
            time.sleep(1)
        else:
            print(f"❌ File not found: {path}")
            time.sleep(2)

    # Initialize Dashboard with recovered steps
    dash = ControlRoom(loaded_name, start_step)

    state = env.reset()
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    while dash.total_frames < MAX_TIMESTEPS:
        if shutdown_flag: break

        # --- PLAY PHASE ---
        for i in range(ROLLOUT_STEPS):
            if shutdown_flag: break

            state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

            t0 = time.perf_counter()
            with torch.no_grad():
                action, log_prob, value = model.get_action(state_tensor)
                confidence = torch.exp(log_prob).item()
            inf_ms = (time.perf_counter() - t0) * 1000

            next_state, reward, done, _ = env.step(action.item())

            state_storage = torch.from_numpy(state).permute(2, 0, 1).cpu()

            states.append(state_storage)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            values.append(value)

            state = next_state
            dash.total_frames += 1
            dash.log_step(inf_ms, confidence)

            if done:
                dash.session_deaths += 1
                state = env.reset()

            if i % 60 == 0:
                dash.render()

        # --- LEARN PHASE ---
        if not shutdown_flag:
            print("\n🔄 Updating Weights...")
            agent.update(states, actions, log_probs, rewards, dones, values)

            states.clear();
            actions.clear();
            log_probs.clear()
            rewards.clear();
            dones.clear();
            values.clear()

            dash.update_count += 1

            if dash.update_count % SAVE_INTERVAL == 0:
                step_str = f"{dash.total_frames // 1000}k"
                filename = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_Step_{step_str}.pt")

                save_checkpoint(model, agent, dash.total_frames, filename)

            dash.render()

    print("\nSaving Exit Checkpoint...")
    filename = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_exit.pt")
    save_checkpoint(model, agent, dash.total_frames, filename)
    sys.exit()


if __name__ == "__main__":
    train()