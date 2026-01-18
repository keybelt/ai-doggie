import torch
import time
import sys
from collections import deque

from env import GeometryDashEnv
from model import GDPolicy, PPOAgent

# --- HYPERPARAMETERS ---
RUN_NAME = "GD_M4_Pro_Pretrain"
MAX_TIMESTEPS = 20_000_000
ROLLOUT_STEPS = 2048
SAVE_INTERVAL = 5


class ControlRoom:
    def __init__(self):
        self.start_time = time.time()
        self.total_frames = 0
        self.update_count = 0

        # Moving Averages for smoothness
        self.latencies = deque(maxlen=100)
        self.confidences = deque(maxlen=100)
        self.fps_buffer = deque(maxlen=100)

        # Trackers
        self.current_attempts = 0
        self.session_best_reward = -999
        self.last_action_type = "IDLE"

    def log_step(self, inference_ms, confidence, fps):
        self.latencies.append(inference_ms)
        self.confidences.append(confidence)
        self.fps_buffer.append(fps)

    def update_dashboard(self, env_deaths, mean_reward, actor_loss, critic_loss):
        elapsed = time.time() - self.start_time
        uptime = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        # Averages
        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        avg_conf = sum(self.confidences) / len(self.confidences) * 100 if self.confidences else 0
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer) if self.fps_buffer else 0

        # ANSI Interface
        sys.stdout.write("\033[H\033[J")  # Clear Screen

        # Status Colors
        lat_color = "\033[92m" if avg_lat < 6.0 else "\033[91m"  # Green if <6ms, Red if >6ms
        conf_color = "\033[96m" if avg_conf > 80 else "\033[93m"  # Cyan if confident, Yellow if unsure
        reset = "\033[0m"

        dashboard = f"""
================================================================================
   GEOMETRY DASH AI | M4 PRO | {RUN_NAME}
================================================================================
 [SYSTEM HEALTH]
  Uptime:       {uptime}
  Frames:       {self.total_frames:,}
  Real FPS:     {avg_fps:.1f} / 120.0
  Queue Depth:  {0} (Ideal: 0-1)

 [BRAIN TELEMETRY]
  Inference:    {lat_color}{avg_lat:.2f} ms{reset} (Target: <8.0ms)
  Confidence:   {conf_color}{avg_conf:.1f}%{reset}
  Model Size:   Small (4-Frame Stack)

 [GAMEPLAY STATS]
  Deaths Detected: {env_deaths} (Check against in-game counter!)
  Avg Reward:      {mean_reward:.4f}
  Best Reward:     {self.session_best_reward:.4f}

 [TRAINING LOSS]
  Actor (Policy):  {actor_loss:.4f}
  Critic (Value):  {critic_loss:.4f}
  Updates:         {self.update_count}
================================================================================
 [LATEST LOG]
 > Collecting batch {self.total_frames % ROLLOUT_STEPS}/{ROLLOUT_STEPS}...
"""
        sys.stdout.write(dashboard)
        sys.stdout.flush()


def train():
    # 1. Setup
    env = GeometryDashEnv()
    device = torch.device("mps")
    model = GDPolicy().to(device)
    agent = PPOAgent(model)
    dash = ControlRoom()

    # Death Counter logic needs to be in Env, but we track it here for now
    # We will approximate deaths by counting 'done' signals
    session_deaths = 0

    state = env.reset()

    # Buffers
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    ep_rewards = deque(maxlen=50)
    current_ep_reward = 0

    try:
        while dash.total_frames < MAX_TIMESTEPS:

            # --- COLLECTION LOOP ---
            for _ in range(ROLLOUT_STEPS):
                loop_start = time.perf_counter()

                # 1. Prepare Input
                state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                # 2. Inference (Measure this!)
                inf_start = time.perf_counter()
                with torch.no_grad():
                    action, log_prob, value = model.get_action(state_tensor)
                    # Calculate confidence (probability of chosen action)
                    probs = torch.exp(log_prob)
                inf_time = (time.perf_counter() - inf_start) * 1000

                # 3. Step
                action_int = action.item()
                next_state, reward, done, _ = env.step(action_int)

                # 4. Record
                states.append(state_tensor.cpu())
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                state = next_state
                current_ep_reward += reward
                dash.total_frames += 1

                # Telemetry Update
                loop_time = time.perf_counter() - loop_start
                current_fps = 1.0 / loop_time if loop_time > 0 else 120
                dash.log_step(inf_time, probs.item(), current_fps)

                if done:
                    session_deaths += 1
                    state = env.reset()
                    ep_rewards.append(current_ep_reward)
                    if current_ep_reward > dash.session_best_reward:
                        dash.session_best_reward = current_ep_reward
                    current_ep_reward = 0

            # --- TRAINING LOOP ---
            avg_rew = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0
            loss_a, loss_c = agent.update(states, actions, log_probs, rewards, dones, values)

            # Clear Buffers
            states.clear();
            actions.clear();
            log_probs.clear();
            rewards.clear();
            dones.clear();
            values.clear()

            dash.update_count += 1
            dash.update_dashboard(session_deaths, avg_rew, loss_a, loss_c)

            if dash.update_count % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f"checkpoints/{RUN_NAME}_{dash.update_count}.pt")

    except KeyboardInterrupt:
        print("\nSaving...")
        torch.save(model.state_dict(), f"checkpoints/{RUN_NAME}_manual_save.pt")
        sys.exit()


if __name__ == "__main__":
    train()