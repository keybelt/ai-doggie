import torch
import time
import sys
import os
import signal
from collections import deque
from env import GeometryDashEnv
from model import GDPolicy, PPOAgent
from pynput.keyboard import Key, Controller

# ... [Config remains same] ...
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RUN_NAME = "Pretrain"
MAX_TIMESTEPS = 20_000_000
ROLLOUT_STEPS = 1024
ACCUMULATION_STEPS = 4
SAVE_INTERVAL = 25
LOAD_CHECKPOINT = None

os_keyboard = Controller()

def safe_pause():
    os_keyboard.press(Key.esc)
    time.sleep(0.05)
    os_keyboard.release(Key.esc)
    time.sleep(0.6)

def safe_resume(env):
    """Resumes the game without wiping the AI's internal memory stack."""
    os_keyboard.press(Key.space)
    time.sleep(0.05)
    os_keyboard.release(Key.space)
    time.sleep(0.1)
    return env.get_state()


class ControlRoom:
    def __init__(self, loaded_from, start_step=0):
        self.start_step_static = start_step
        self.total_frames = start_step
        self.update_count = 0
        self.loaded_from = loaded_from if loaded_from else "Fresh Start"
        self.latencies = deque(maxlen=20)
        self.confidences = deque(maxlen=20)
        self.total_drops = 0
        self.current_entropy = 0.0
        self.current_policy_loss = 0.0
        self.session_deaths = 0
        self.status_message = "Active"

    def log_step(self, inference_ms, confidence, drops):
        self.latencies.append(inference_ms)
        self.confidences.append(confidence)
        self.total_drops = drops  # Update tracking

    def log_update(self, policy_loss, entropy):
        self.current_policy_loss = policy_loss
        self.current_entropy = entropy
        self.update_count += 1

    def render(self):
        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_conf = sum(self.confidences) / len(self.confidences) * 100 if self.confidences else 0
        c_reset, c_green, c_yellow, c_red, c_cyan = "\033[0m", "\033[92m", "\033[93m", "\033[91m", "\033[96m"
        lat_color = c_green if avg_lat < 8.33 else c_red
        drop_color = c_green if self.total_drops == 0 else c_red
        ent_color = c_yellow if self.current_entropy > 0.4 else c_green
        loss_color = c_red if abs(self.current_policy_loss) > 2.0 else c_cyan
        progress = self.update_count % SAVE_INTERVAL
        bar = "█" * int((progress / SAVE_INTERVAL) * 10) + "░" * (10 - int((progress / SAVE_INTERVAL) * 10))

        os.system('clear')
        print(f"""
============================================================
   AI Zoink | {RUN_NAME}
============================================================
 [SOURCE]
  > Loaded From:       {c_cyan}{self.loaded_from}{c_reset}
  > Starting Step:     {self.start_step_static:,}

 [LIVE PERFORMANCE]
  > Inference Latency: {lat_color}{avg_lat:.2f} ms{c_reset}
  > AI Confidence:     {c_green}{avg_conf:.1f}%{c_reset}
  > Capture Drops:     {drop_color}{self.total_drops}{c_reset}
  > Session Deaths:    {c_red}{self.session_deaths}{c_reset}

 [RL BRAIN METRICS]
  > Curiosity (Ent):   {ent_color}{self.current_entropy:.4f}{c_reset}
  > Policy Loss:       {loss_color}{self.current_policy_loss:.4f}{c_reset}

 [TRAINING PROGRESS]
  > Global Steps:      {c_cyan}{self.total_frames:,}{c_reset}
  > Weight Updates:    {self.update_count}
  > Next Checkpoint:   [{bar}] {progress}/{SAVE_INTERVAL}

 [STATUS]
  > {c_yellow}{self.status_message}{c_reset}
============================================================
""")


shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)


def save_checkpoint(model, agent, step_count, filename):
    torch.save(
        {'model_state': model.state_dict(), 'optimizer_state': agent.optimizer.state_dict(), 'global_step': step_count},
        filename)


def train():
    global shutdown_flag
    print("⚠️  FOCUS GEOMETRY DASH NOW...")
    time.sleep(5)

    env = GeometryDashEnv()
    device = torch.device("mps")
    model = GDPolicy().to(device)
    agent = PPOAgent(model)

    # Load Logic
    start_step = 0
    loaded_name = None
    if LOAD_CHECKPOINT:
        path = os.path.join(CHECKPOINT_DIR, LOAD_CHECKPOINT)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_step = checkpoint['global_step']
            loaded_name = LOAD_CHECKPOINT

    dash = ControlRoom(loaded_name, start_step)
    state = env.reset()
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    while dash.total_frames < MAX_TIMESTEPS:
        if shutdown_flag: break

        for accum_step in range(ACCUMULATION_STEPS):
            if shutdown_flag: break
            dash.status_message = f"Collecting Experience ({accum_step + 1}/{ACCUMULATION_STEPS})..."

            for i in range(ROLLOUT_STEPS):
                if shutdown_flag: break

                # Ensure the dashboard renders regularly to prevent freezing appearance
                if i % 50 == 0: dash.render()

                state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                t_compute_start = time.perf_counter()
                with torch.no_grad():
                    action, log_prob, value = model.get_action(state_tensor)
                    confidence = torch.exp(log_prob).item()
                inference_time = (time.perf_counter() - t_compute_start) * 1000

                next_state, reward, done, info = env.step(action.item())

                # Handle internal warnings from env without stopping the whole loop
                if "warning" in info:
                    continue

                states.append(torch.from_numpy(state).permute(2, 0, 1).cpu())
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = next_state
                dash.total_frames += 1
                drops = info.get("drops")
                dash.log_step(inference_time, confidence, drops)

                if done:
                    dash.session_deaths += 1
                    state = env.reset()

        if not shutdown_flag:
            dash.status_message = "⏸ Updating Weights..."
            dash.render()

            safe_pause()
            env.engine.stop_stream()

            loss, entropy = agent.update(states, actions, log_probs, rewards, dones, values)
            dash.log_update(loss, entropy)

            states.clear();
            actions.clear();
            log_probs.clear()
            rewards.clear();
            dones.clear();
            values.clear()

            if dash.update_count % SAVE_INTERVAL == 0:
                save_checkpoint(model, agent, dash.total_frames,
                                os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_Step_{dash.total_frames // 1000}k.pt"))

            dash.status_message = "🔄 Reconnecting Vision..."
            dash.render()
            env.engine.resume_stream()

            time.sleep(0.5)
            env.flush_vision()

            dash.status_message = "▶️ Resuming Gameplay..."
            dash.render()
            state = safe_resume(env)

    save_checkpoint(model, agent, dash.total_frames, os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_exit.pt"))
    sys.exit()


if __name__ == "__main__":
    train()