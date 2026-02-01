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

    def act(self, action):
        if action == 1:
            if not self.is_holding:
                self._post_event(True)
                self.is_holding = True
                return True
        else:
            if self.is_holding:
                self._post_event(False)
                self.is_holding = False
        return False

    def _post_event(self, key_down):
        event = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, key_down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


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
        self.attempt_roi = (56, 3, 53, 13)
        self.lower_spectrum = np.array([70, 100, 100])
        self.upper_spectrum = np.array([80, 255, 255])

        self.frame_stack = collections.deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

        self.last_color_mask = None
        self.steps_since_reset = 0
        self.state_buffer = np.zeros((332, 588, 12), dtype=np.uint8)

    def reset(self):
        self.controller.act(0)
        self.last_color_mask = None
        self.steps_since_reset = 0

        self.frame_stack.clear()

        _ = self.flush_vision()

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)

            initial_frame = raw_frame[:, :, :3].copy()

            for _ in range(self.stack_size):
                self.frame_stack.append(initial_frame)

            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put(raw_frame)
        except:
            print("[Env] Warning: Reset timed out waiting for fresh frame.")
            for _ in range(self.stack_size):
                self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

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
        self.controller.act(action)
        self.steps_since_reset += 1

        lag_skips = self.flush_vision()
        self.cumulative_lag_skips += lag_skips

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=0.1)
        except:
            return self.get_state(), 0.0, False, {"warning": "frame_skip"}

        current_frame = raw_frame[:, :, :3].copy()

        if hasattr(self.engine, 'idle_queue'):
            self.engine.idle_queue.put(raw_frame)

        tx, ty, tw, th = self.attempt_roi
        text_crop = current_frame[ty:ty + th, tx:tx + tw]

        hsv_crop = cv2.cvtColor(text_crop, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_crop, self.lower_spectrum, self.upper_spectrum)

        is_dead = False
        if self.last_color_mask is not None:
            mask_diff = cv2.bitwise_xor(color_mask, self.last_color_mask)
            if cv2.countNonZero(mask_diff) > 3:
                is_dead = True

        self.last_color_mask = color_mask.copy()
        self.frame_stack.append(current_frame)

        if is_dead:
            reward = -50.0
            done = True
        else:
            reward = 0.3
            done = False

        perf_misses = self.engine.drop_count + self.cumulative_lag_skips

        return self.get_state(), reward, done, {"missed": perf_misses}

    def resume_session(self):
        _ = self.flush_vision()
        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)
            self.frame_stack.append(raw_frame[:, :, :3].copy())

            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put(raw_frame)
        except:
            print("[Env] Warning: Resume timed out waiting for frame.")

        return self.get_state()


_ = NSApplication.sharedApplication()