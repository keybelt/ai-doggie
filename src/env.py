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

        self.cumulative_lag_skips = 0

        self.flush_vision()
        self.attempt_roi = (279, 1, 28, 16)

        self.death_template = cv2.imread('0%.png', cv2.IMREAD_GRAYSCALE)
        if self.death_template is None:
            raise ValueError("Could not load 0%.png")

        self.frame_stack = collections.deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

        self.steps_since_reset = 0
        self.state_buffer = np.zeros((332, 588, 12), dtype=np.uint8)

    def reset(self):
        self.controller.act(0)
        self.engine.paused = True
        self.last_color_mask = None
        self.steps_since_reset = 0

        self.frame_stack.clear()

        lag_skips = self.flush_vision()
        self.cumulative_lag_skips += lag_skips

        try:
            self.engine.paused = False
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)
            initial_frame = raw_frame[:, :, :3].copy()

            for _ in range(self.stack_size):
                self.frame_stack.append(initial_frame)

            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put_nowait(raw_frame)
        except:
            print("[Env] Warning: Reset timed out waiting for fresh frame.")
            for _ in range(self.stack_size):
                self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

        post_wait_skips = self.flush_vision()
        self.cumulative_lag_skips += post_wait_skips

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
        for i, frame in enumerate(self.frame_stack):
            self.state_buffer[:, :, i * 3:(i + 1) * 3] = frame
        return self.state_buffer.copy()

    def step(self, action):
        did_initiate_press = self.controller.act(action)
        self.steps_since_reset += 1

        lag_skips = self.flush_vision()
        self.cumulative_lag_skips += lag_skips

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=0.1)
        except:
            return self.get_state(), 0.0, False, {"warning": "frame_skip"}

        current_frame = raw_frame[:, :, :3].copy()

        if hasattr(self.engine, 'idle_queue'):
            self.engine.idle_queue.put_nowait(raw_frame)

        tx, ty, tw, th = self.attempt_roi
        text_crop = current_frame[ty:ty + th, tx:tx + tw]

        gray_crop = cv2.cvtColor(text_crop, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray_crop, self.death_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        is_dead = max_val > 0.9
        self.frame_stack.append(current_frame)

        if is_dead:
            reward = -50.0
            done = True
        else:
            reward = 0.1
            if did_initiate_press:
                reward -= 0.01
            done = False

        perf_misses = self.engine.drop_count + self.cumulative_lag_skips

        return self.get_state(), reward, done, {"missed": perf_misses}

    def resume_session(self):
        _ = self.flush_vision()
        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)
            self.frame_stack.append(raw_frame[:, :, :3].copy())

            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put_nowait(raw_frame)
        except:
            print("[Env] Warning: Resume timed out waiting for frame.")

        return self.get_state()


_ = NSApplication.sharedApplication()