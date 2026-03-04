import numpy as np
import collections
import cv2
import sys
import Quartz
from AppKit import NSApplication
from capture import start_capture


class KeyboardController:
    def __init__(self):
        self.is_holding = False
        self.SPACE_KEY = 49
        self.src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)

        self.ev_down = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, True)
        self.ev_up = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, False)

    def act(self, action):
        if action == 1:
            if not self.is_holding:
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, self.ev_down)
                self.is_holding = True
                return True
        else:
            if self.is_holding:
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, self.ev_up)
                self.is_holding = False
        return False


class GeometryDashEnv:
    def __init__(self):
        self.engine = start_capture()
        self.controller = KeyboardController()

        print("[Env] Connecting to Vision Engine...")
        self.engine.paused = False

        try:
            self.engine.ready_queue.get(timeout=10.0)
            print("[Env] Vision Connected!")
        except:
            print("\n[CRITICAL ERROR] Vision Engine timed out!")
            sys.exit(1)

        self.stack_size = 4

        self.dilation = 4

        self.raw_buffer_size = (self.stack_size - 1) * self.dilation + 1
        self.raw_frame_buffer = collections.deque(maxlen=self.raw_buffer_size)
        for _ in range(self.raw_buffer_size):
            self.raw_frame_buffer.append(np.zeros((332, 588, 3), dtype=np.uint8))

        self.state_buffer = np.zeros((332, 588, 12), dtype=np.uint8)

        self.cumulative_lag_skips = 0

        self.flush_vision()
        self.attempt_roi = (264, 1, 30, 16)

        self.death_template = cv2.imread('0%.png', cv2.IMREAD_GRAYSCALE)
        if self.death_template is None:
            raise ValueError("Could not load 0%.png")

        self.steps_since_reset = 0
        self.state_buffer = np.zeros((332, 588, 12), dtype=np.uint8)
        self.last_template_match = False
        self.min_hold_frames = 1
        self.current_hold_steps = 0
        self.last_action = 0
        self.death_cooldown_duration = 30
        self.current_death_cooldown = 0

    def reset(self):
        self.controller.act(0)
        self.engine.paused = True
        self.last_color_mask = None
        self.steps_since_reset = 0
        self.current_hold_steps = 0
        self.last_action = 0
        self.current_death_cooldown = 0

        lag_skips = self.flush_vision()
        self.cumulative_lag_skips += lag_skips

        try:
            self.engine.paused = False
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)
            initial_frame = raw_frame[:, :, :3].copy()

            tx, ty, tw, th = self.attempt_roi
            init_crop = initial_frame[ty:ty + th, tx:tx + tw]
            init_gray = cv2.cvtColor(init_crop, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(init_gray, self.death_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            self.last_template_match = max_val >= 0.99995

            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put_nowait(raw_frame)
        except:
            print("[Env] Warning: Reset timed out waiting for fresh frame.")

        post_wait_skips = self.flush_vision()
        self.cumulative_lag_skips += post_wait_skips

        self.initial_drop_count = self.engine.drop_count

        self.raw_frame_buffer.clear()

        for _ in range(self.raw_buffer_size):
            self.raw_frame_buffer.append(initial_frame)

        return self.get_state()

    def flush_vision(self):
        if self.engine.ready_queue.empty():
            return 0

        skipped_in_batch = 0
        while self.engine.ready_queue.qsize() > 1:
            try:
                frame, _ = self.engine.ready_queue.get_nowait()
                if hasattr(self.engine, 'idle_queue'):
                    self.engine.idle_queue.put(frame)
                skipped_in_batch += 1
            except:
                break

        return skipped_in_batch

    def get_state(self):
        for i in range(self.stack_size):
            idx = -1 - (i * self.dilation)
            frame = self.raw_frame_buffer[idx]

            self.state_buffer[:, :, i * 3:(i + 1) * 3] = frame

        return self.state_buffer.copy()

    def step(self, action):
        if self.current_hold_steps > 0:
            action = 1
            self.current_hold_steps -= 1
        elif action == 1:
            self.current_hold_steps = self.min_hold_frames - 1

        did_initiate_press = self.controller.act(action)
        self.steps_since_reset += 1

        lag_skips = self.flush_vision()
        self.cumulative_lag_skips += lag_skips

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=0.1)
        except:
            return self.get_state(), 0.0, False, {"warning": "frame_skip"}

        current_frame = raw_frame[:, :, :3].copy()

        self.raw_frame_buffer.append(current_frame)

        if hasattr(self.engine, 'idle_queue'):
            self.engine.idle_queue.put_nowait(raw_frame)

        tx, ty, tw, th = self.attempt_roi
        text_crop = current_frame[ty:ty + th, tx:tx + tw]

        gray_crop = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray_crop, self.death_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        matches_template = max_val >= 0.99995
        is_dead = False
        if self.current_death_cooldown > 0:
            self.current_death_cooldown -= 1
            self.last_template_match = matches_template
        else:
            is_dead = matches_template and not self.last_template_match
            self.last_template_match = matches_template

            if is_dead:
                self.current_death_cooldown = self.death_cooldown_duration

        if is_dead:
            reward = -5.0
            done = True
        else:
            reward = 0.01
            if action != self.last_action:
                reward -= 0.1
            self.last_action = action
            done = False

        episode_drops = self.engine.drop_count - self.initial_drop_count
        perf_misses = episode_drops + self.cumulative_lag_skips

        return self.get_state(), reward, done, {"missed": perf_misses}

    def resume_session(self):
        _ = self.flush_vision()
        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)
            self.raw_frame_buffer.append(raw_frame[:, :, :3].copy())

            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put_nowait(raw_frame)
        except:
            print("[Env] Warning: Resume timed out waiting for frame.")

        return self.get_state()


_ = NSApplication.sharedApplication()