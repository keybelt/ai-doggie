import torch
import time
import sys
import os
import signal
from collections import deque
from env import GeometryDashEnv
from model import GDPolicy, PPOAgent
from pynput.keyboard import Key, Controller

# Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RUN_NAME = "Pretrain"
MAX_TIMESTEPS = 20_000_000
ROLLOUT_STEPS = 4096
ACCUMULATION_STEPS = 1
SAVE_INTERVAL = 100
LOAD_CHECKPOINT = None

os_keyboard = Controller()

def safe_pause():
    os_keyboard.press(Key.esc)
    time.sleep(0.05)
    os_keyboard.release(Key.esc)
    time.sleep(0.02)


def safe_resume(env):
    env.flush_vision()
    # Press space to unpause (assuming Space unpauses in your config)
    os_keyboard.press(Key.space)
    time.sleep(0.05)
    os_keyboard.release(Key.space)
    time.sleep(0.02)

    # CHANGED: Use resume_session() instead of reset()
    # This prevents the "Black Frame" blindness after updates
    return env.resume_session()


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
        self.total_drops = drops

    def log_update(self, policy_loss, entropy):
        self.current_policy_loss = policy_loss
        self.current_entropy = entropy
        self.update_count += 1

    def render(self):
        c_reset, c_green, c_yellow, c_red = "\033[0m", "\033[92m", "\033[93m", "\033[91m"
        progress = self.update_count % SAVE_INTERVAL

        avg_conf = sum(self.confidences) / len(self.confidences) * 100 if self.confidences else 0
        conf_color = c_green if avg_conf > 85 else c_yellow if avg_conf > 65 else c_red

        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        lat_color = c_green if avg_lat < 6.0 else c_yellow if avg_lat < 8.0 else c_red

        total_generated = (self.total_drops + self.total_frames)
        drop_pct = (self.total_drops / total_generated * 100) if total_generated > 0 else 0
        drop_color = c_green if drop_pct == 0 else (c_yellow if drop_pct < 0.05 else c_red)

        ent_color = c_green if self.current_entropy > 0.4 else c_yellow if self.current_entropy > 0.15 else c_red
        ent_label = "EXPLORING" if self.current_entropy > 0.4 else "LEARNING" if self.current_entropy > 0.15 else "STUCK"

        loss_color = c_green if abs(self.current_policy_loss) < 0.02 else c_yellow if abs(self.current_policy_loss) < 0.05 else c_red
        loss_label = "HEALTHY" if abs(self.current_policy_loss) < 0.02 else "WARNING" if abs(self.current_policy_loss) < 0.05 else "CRITICAL"

        os.system('clear')
        print(f"""
============================================================
   AI Zoink | {RUN_NAME}
============================================================
 [SOURCE]
  > Loaded From:       {self.loaded_from}
  > Starting Step:     {self.start_step_static:,}

 [LIVE PERFORMANCE]
  > Inference Latency: {lat_color}{avg_lat:.2f} ms{c_reset}
  > AI Confidence:     {conf_color}{avg_conf:.1f}%{c_reset}
  > Capture Drops:     {self.total_drops} ({drop_color}{drop_pct:.3f}%{c_reset})
  > Session Deaths:    {self.session_deaths}

 [RL BRAIN METRICS]
  > Curiosity (Ent):   {self.current_entropy:.3f} ({ent_color}{ent_label}{c_reset})
  > Policy Loss:       {self.current_policy_loss:.3f} ({loss_color}{loss_label}{c_reset})

 [TRAINING PROGRESS]
  > Global Steps:      {self.total_frames:,}
  > Weight Updates:    {self.update_count}
  > Next Checkpoint:   {progress}/{SAVE_INTERVAL}

 [STATUS]
  > {self.status_message}
============================================================
""")


shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)


def save_checkpoint(model, agent, step_count, filename):
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'global_step': step_count
    }, filename)

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
                if i % 50 == 0: dash.render()

                state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                t_start = time.perf_counter()
                with torch.no_grad():
                    action, log_prob, value = model.get_action(state_tensor)
                    confidence = torch.exp(log_prob).item()
                inf_ms = (time.perf_counter() - t_start) * 1000

                next_state, reward, done, info = env.step(action.item())

                if "warning" in info: continue

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = next_state
                dash.total_frames += 1
                dash.log_step(inf_ms, confidence, info.get("drops"))

                if done:
                    dash.session_deaths += 1
                    state = env.reset()

        if not shutdown_flag:
            dash.status_message = "⏸️ Updating Weights..."
            dash.render()
            safe_pause()

            # 1. Snapshot drop count BEFORE update
            drops_before_update = env.engine.drop_count

            # 2. Perform the heavy update (blocking the queue consumer)
            loss, entropy = agent.update(states, actions, log_probs, rewards, dones, values)

            # 3. Snapshot drop count AFTER update
            drops_after_update = env.engine.drop_count

            # 4. Calculate "Maintenance Drops" that shouldn't count towards performance
            ignored_drops = drops_after_update - drops_before_update

            # 5. Subtract these from the dashboard's total so the metric stays clean
            dash.total_drops -= ignored_drops

            dash.log_update(loss, entropy)

            states.clear(); actions.clear(); log_probs.clear()
            rewards.clear(); dones.clear(); values.clear()

            if dash.update_count % SAVE_INTERVAL == 0:
                dash.status_message = "💾 SAVING CHECKPOINT..."
                save_checkpoint(model, agent, dash.total_frames, os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_S{dash.total_frames//1000}k.pt"))

            dash.status_message = "▶️ Resuming Gameplay..."
            dash.render()
            state = safe_resume(env)

    save_checkpoint(model, agent, dash.total_frames, os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_exit.pt"))
    sys.exit()


if __name__ == "__main__":
    train()