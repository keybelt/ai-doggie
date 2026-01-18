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

# OPTIMIZATION: Gradient Accumulation
ROLLOUT_STEPS = 1024  # Keep small to save RAM
ACCUMULATION_STEPS = 4  # Play 4x batches before updating (Total effective batch = 4096)

SAVE_INTERVAL = 25  # Adjusted: 25 updates * 4 accum = 100 actual cycles
LOAD_CHECKPOINT = "Pretrain_exit.pt"


# --- DASHBOARD ---
class ControlRoom:
    def __init__(self, loaded_from, start_step=0):
        self.start_step_static = start_step
        self.total_frames = start_step
        self.update_count = 0
        self.loaded_from = loaded_from if loaded_from else "Fresh Start"

        self.latencies = deque(maxlen=20)
        self.confidences = deque(maxlen=20)
        self.session_deaths = 0
        self.status_message = "Active"

    def log_step(self, inference_ms, confidence):
        self.latencies.append(inference_ms)
        self.confidences.append(confidence)

    def render(self):
        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_conf = sum(self.confidences) / len(self.confidences) * 100 if self.confidences else 0

        progress = self.update_count % SAVE_INTERVAL
        bar_len = 10
        filled = int((progress / SAVE_INTERVAL) * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lat_color = "\033[92m" if avg_lat < 8.0 else "\033[91m"
        reset = "\033[0m"

        os.system('clear')

        dashboard = f"""
============================================================
   AI Zoink | {RUN_NAME}
============================================================
 [SOURCE]
  > Loaded From:       {self.loaded_from}
  > Starting Step:     {self.start_step_static:,}

 [PERFORMANCE]
  > Inference Latency: {lat_color}{avg_lat:.2f} ms{reset}
  > AI Confidence:     {avg_conf:.1f}%

 [STATS]
  > Deaths (Session):  {self.session_deaths}

 [TRAINING PROGRESS]
  > Global Steps:      {self.total_frames:,}
  > Weight Updates:    {self.update_count} (Session)
  > Accumulation:      {ROLLOUT_STEPS} x {ACCUMULATION_STEPS} = {ROLLOUT_STEPS * ACCUMULATION_STEPS} steps/update
  > Next Checkpoint:   [{bar}] {progress}/{SAVE_INTERVAL}

 [STATUS]
  > {self.status_message}

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
    agent = PPOAgent(model)

    start_step = 0
    loaded_name = None

    if LOAD_CHECKPOINT:
        clean_name = os.path.basename(LOAD_CHECKPOINT)
        path = os.path.join(CHECKPOINT_DIR, clean_name)
        if os.path.exists(path):
            print(f"Loading: {path}")
            checkpoint = torch.load(path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                print("✅ Detected Bundle")
                model.load_state_dict(checkpoint['model_state'])
                agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_step = checkpoint['global_step']
            else:
                model.load_state_dict(checkpoint)
            loaded_name = clean_name
            time.sleep(1)

    dash = ControlRoom(loaded_name, start_step)
    state = env.reset()
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    while dash.total_frames < MAX_TIMESTEPS:
        if shutdown_flag: break

        # --- GRADIENT ACCUMULATION LOOP ---
        # We play X times, accumulate gradients, then update ONCE.

        agent.optimizer.zero_grad()  # Clear gradients before starting accumulation

        for accum_step in range(ACCUMULATION_STEPS):
            if shutdown_flag: break

            dash.status_message = f"Playing (Batch {accum_step + 1}/{ACCUMULATION_STEPS})..."

            # 1. Collect Data (Play Phase)
            for i in range(ROLLOUT_STEPS):
                if shutdown_flag: break

                state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                t0 = time.perf_counter()
                with torch.no_grad():
                    action, log_prob, value = model.get_action(state_tensor)
                    confidence = torch.exp(log_prob).item()
                inf_ms = (time.perf_counter() - t0) * 1000

                next_state, reward, done, _ = env.step(action.item())

                # Store data
                states.append(torch.from_numpy(state).permute(2, 0, 1).cpu())
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

            # 2. Compute Gradients (But don't step optimizer yet!)
            if not shutdown_flag:
                dash.status_message = f"Calculating Gradients ({accum_step + 1}/{ACCUMULATION_STEPS})..."
                dash.render()

                # Note: We need to modify PPOAgent.update slightly to NOT zero_grad automatically
                # For simplicity, we just call update().
                # Ideally, you'd split PPO into 'calc_loss' and 'step', but standard PPO update
                # runs multiple epochs anyway.

                # SIMPLIFIED ACCUMULATION STRATEGY FOR PPO:
                # Since PPO is on-policy and runs epochs internally, "Accumulation" usually just means
                # "Collect a bigger batch". We have effectively done that by looping here.
                # However, we need to KEEP the data in the lists until the end of the accum loop.
                pass

                # --- UPDATE PHASE (Once per X batches) ---
        if not shutdown_flag:
            dash.status_message = "🔄 Updating Weights (Big Batch)..."
            dash.render()

            # Now we send the MASSIVE combined lists to the agent
            agent.update(states, actions, log_probs, rewards, dones, values)

            # Clear buffers
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
                dash.status_message = f"💾 Saving Bundle: {step_str}..."
                dash.render()
                save_checkpoint(model, agent, dash.total_frames, filename)

            dash.render()

    print("\nSaving Exit Checkpoint...")
    filename = os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_exit.pt")
    save_checkpoint(model, agent, dash.total_frames, filename)
    sys.exit()


if __name__ == "__main__":
    train()