import torch
import time
import sys
import os
import signal
import gc
from collections import deque
from env import GeometryDashEnv
from model import GDPolicy, PPOAgent
import Quartz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "checkpoints"))
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

RUN_NAME = "Pretrain"
MAX_TIMESTEPS = 20_000_000
ROLLOUT_STEPS = 2048
ACCUMULATION_STEPS = 1
SAVE_INTERVAL = 100
LOAD_CHECKPOINT = "Pretrain_exit.pt"


def send_global_key(keycode):
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    ev_down = Quartz.CGEventCreateKeyboardEvent(src, keycode, True)
    ev_up = Quartz.CGEventCreateKeyboardEvent(src, keycode, False)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
    time.sleep(0.05)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)


def safe_pause():
    send_global_key(53)
    time.sleep(0.02)


def safe_resume(env):
    start_drops = env.engine.drop_count
    start_skips = env.skipped_count

    env.engine.paused = False

    wait_start = time.time()
    while env.engine.ready_queue.empty():
        if time.time() - wait_start > 2.0:
            print("[WARN] Resume timed out waiting for queue")
            break
        time.sleep(0.001)

    send_global_key(49)

    state = env.resume_session()

    end_drops = env.engine.drop_count
    end_skips = env.skipped_count

    delta_drops = end_drops - start_drops
    delta_skips = end_skips - start_skips

    return state, delta_drops, delta_skips


class ControlRoom:
    def __init__(self, loaded_from, start_step=0):
        self.start_step_static = start_step
        self.total_frames = start_step
        self.update_count = 0
        self.loaded_from = loaded_from if loaded_from else "Fresh Start"
        self.latencies = deque(maxlen=20)
        self.confidences = deque(maxlen=20)
        self.total_missed = 0
        self.current_entropy = 0.0
        self.current_policy_loss = 0.0
        self.session_deaths = 0
        self.status_message = "Active"

    def log_step(self, inference_ms, confidence, missed_frames):
        self.latencies.append(inference_ms)
        self.confidences.append(confidence)
        self.total_missed = missed_frames

    def log_update(self, policy_loss, entropy):
        self.current_policy_loss = policy_loss
        self.current_entropy = entropy
        self.update_count += 1

    def render(self):
        c_reset, c_green, c_yellow, c_red = "\033[0m", "\033[92m", "\033[93m", "\033[91m"
        progress = self.update_count % SAVE_INTERVAL

        avg_conf = sum(self.confidences) / len(self.confidences) * 100 if self.confidences else 0
        conf_color = c_green if avg_conf > 90 else c_yellow if avg_conf > 75 else c_red

        avg_lat = sum(self.latencies) / len(self.latencies) if self.latencies else 0
        lat_color = c_green if avg_lat < 4.0 else c_yellow if avg_lat < 6.0 else c_red

        session_frames = self.total_frames - self.start_step_static
        miss_pct = (self.total_missed / (session_frames + 1e-7) * 100)
        miss_color = c_green if miss_pct == 0 else (c_yellow if miss_pct < 0.05 else c_red)

        ent_color = c_green if self.current_entropy > 0.5 else c_yellow if self.current_entropy > 0.1 else c_red
        ent_label = "EXPLORING" if self.current_entropy > 0.5 else "CONVERGING" if self.current_entropy > 0.1 else "DETERMINISTIC"

        loss_val = abs(self.current_policy_loss)
        loss_color = c_green if loss_val < 0.05 else c_yellow if loss_val < 0.15 else c_red
        loss_label = "STABLE" if loss_val < 0.05 else "VOLATILE" if loss_val < 0.15 else "UNSTABLE"

        print(f"""\033[H\033[J""")
        print(f"""
============================================================
   AI Doggie | {RUN_NAME}
============================================================
 [SOURCE]
  > Loaded From:       {self.loaded_from}
  > Starting Step:     {self.start_step_static:,}

 [LIVE PERFORMANCE]
  > Inference Latency: {lat_color}{avg_lat:.2f} ms{c_reset}
  > AI Confidence:     {conf_color}{avg_conf:.1f}%{c_reset}
  > Missed Frames:     {self.total_missed} ({miss_color}{miss_pct:.3f}%{c_reset})
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


class RolloutBuffer:
    def __init__(self, steps, height, width, channels):
        self.steps = steps
        self.states = torch.zeros((steps, height, width, channels), dtype=torch.uint8)
        self.actions = torch.zeros(steps, dtype=torch.long)
        self.log_probs = torch.zeros(steps, dtype=torch.float)
        self.rewards = torch.zeros(steps, dtype=torch.float)
        self.dones = torch.zeros(steps, dtype=torch.bool)
        self.values = torch.zeros(steps, dtype=torch.float)
        self.ptr = 0

    def add(self, state, action, log_prob, reward, done, value):
        if self.ptr < self.steps:
            self.states[self.ptr] = torch.from_numpy(state)
            self.actions[self.ptr] = action
            self.log_probs[self.ptr] = log_prob
            self.rewards[self.ptr] = reward
            self.dones[self.ptr] = done
            self.values[self.ptr] = value
            self.ptr += 1

    def get(self):
        self.ptr = 0
        return self.states, self.actions, self.log_probs, self.rewards, self.dones, self.values


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

    print("[System] Allocating Memory Buffer...")
    buffer = RolloutBuffer(ROLLOUT_STEPS * ACCUMULATION_STEPS, 332, 588, 12)

    env.engine.paused = False

    print("WARMING UP VISION & MODEL (120 Frames + Compilation)...")

    dummy_state = torch.zeros((1, 12, 332, 588)).to(device)
    state = env.reset()

    for _ in range(120):
        _ = env.step(0)
        with torch.no_grad():
            _ = model.get_action(dummy_state)

    env.flush_vision()
    env.engine.drop_count = 0
    env.skipped_count = 0

    ignore_drops = 0
    ignore_skips = 0

    print("Startup complete. Metrics forcibly reset to 0.")

    dash = ControlRoom(loaded_name, start_step)

    while dash.total_frames < MAX_TIMESTEPS:
        if shutdown_flag: break

        for accum_step in range(ACCUMULATION_STEPS):
            if shutdown_flag: break

            gc.disable()

            dash.status_message = f"Collecting Experience ({accum_step + 1}/{ACCUMULATION_STEPS})..."

            for i in range(ROLLOUT_STEPS):
                if shutdown_flag: break
                if i % 100 == 0: dash.render()

                t_start = time.perf_counter()

                state_tensor = torch.from_numpy(state).to(device)
                state_tensor = state_tensor.float().div(255.0)
                state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

                with torch.no_grad():
                    action, log_prob, value = model.get_action(state_tensor)
                    confidence = torch.exp(log_prob).item()

                inf_ms = (time.perf_counter() - t_start) * 1000

                next_state, reward, done, info = env.step(action.item())

                net_drops = env.engine.drop_count - ignore_drops
                net_skips = env.skipped_count - ignore_skips
                current_session_missed = net_drops + net_skips

                #if i % 100 == 0:
                #    print(
                #        f"[RAW DEBUG] Drops: {env.engine.drop_count} | Skips: {env.skipped_count} | Queue: {env.engine.ready_queue.qsize()}")

                dash.log_step(inf_ms, confidence, current_session_missed)
                if "warning" in info:
                    state = next_state
                    continue

                buffer.add(state, action.item(), log_prob, reward, done, value)

                state = next_state
                dash.total_frames += 1

                if done:
                    dash.session_deaths += 1
                    state = env.reset()

            gc.enable()

        if not shutdown_flag:
            dash.status_message = "⏸️ Updating Weights..."
            dash.render()
            env.engine.paused = True
            gc.collect()
            safe_pause()

            b_states, b_actions, b_log_probs, b_rewards, b_dones, b_values = buffer.get()
            loss, entropy = agent.update(b_states, b_actions, b_log_probs, b_rewards, b_dones, b_values)

            dash.log_update(loss, entropy)

            if dash.update_count % SAVE_INTERVAL == 0:
                dash.status_message = "💾 SAVING CHECKPOINT..."
                save_checkpoint(model, agent, dash.total_frames,
                                os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_{dash.total_frames // 1000}k.pt"))

            dash.status_message = "▶️ Resuming Gameplay..."
            dash.render()
            env.flush_vision()

            state, d_drops, d_skips = safe_resume(env)

            ignore_drops += d_drops
            ignore_skips += d_skips

    save_checkpoint(model, agent, dash.total_frames, os.path.join(CHECKPOINT_DIR, f"{RUN_NAME}_exit.pt"))
    sys.exit()


if __name__ == "__main__":
    train()