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

        self.skipped_count = 0

        self.flush_vision()
        self.attempt_roi = (56, 3, 53, 13)
        self.lower_spectrum = np.array([70, 100, 100])
        self.upper_spectrum = np.array([80, 255, 255])

        self.frame_stack = collections.deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

        self.last_color_mask = None
        self.steps_since_reset = 0

    def reset(self):
        self.controller.act(0)
        self.last_color_mask = None
        self.steps_since_reset = 0

        self.frame_stack.clear()

        self.flush_vision()

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)

            initial_frame = raw_frame.copy()

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
            return

        skipped_in_batch = 0
        while self.engine.ready_queue.qsize() > 1:
            try:
                frame, _ = self.engine.ready_queue.get_nowait()
                if hasattr(self.engine, 'idle_queue'):
                    self.engine.idle_queue.put(frame)
                skipped_in_batch += 1
            except:
                break

        self.skipped_count += skipped_in_batch

    def get_state(self):
        return np.concatenate(list(self.frame_stack), axis=2)

    def step(self, action):
        self.controller.act(action)
        self.steps_since_reset += 1

        self.flush_vision()

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=0.1)
        except:
            return self.get_state(), 0.0, False, {"warning": "frame_skip"}

        current_frame = raw_frame.copy()
        if hasattr(self.engine, 'idle_queue'):
            self.engine.idle_queue.put(raw_frame)

        tx, ty, tw, th = self.attempt_roi
        text_crop = current_frame[ty:ty + th, tx:tx + tw]
        hsv_crop = cv2.cvtColor(text_crop, cv2.COLOR_RGB2HSV)
        color_mask = cv2.inRange(hsv_crop, self.lower_spectrum, self.upper_spectrum)

        is_dead = False
        if self.last_color_mask is not None:
            mask_diff = cv2.bitwise_xor(color_mask, self.last_color_mask)
            if cv2.countNonZero(mask_diff) > 5:
                is_dead = True

        self.last_color_mask = color_mask.copy()
        self.frame_stack.append(current_frame)

        if is_dead:
            reward = -100.0
            done = True
        else:
            reward = 0.1
            done = False

        total_missed = self.engine.drop_count + self.skipped_count

        return self.get_state(), reward, done, {"missed": total_missed}

    def resume_session(self):
        self.flush_vision()
        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=2.0)
            self.frame_stack.append(raw_frame)
            if hasattr(self.engine, 'idle_queue'):
                self.engine.idle_queue.put(raw_frame)
        except:
            print("[Env] Warning: Resume timed out waiting for frame.")

        return self.get_state()


_ = NSApplication.sharedApplication()