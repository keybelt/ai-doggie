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
        try:
            self.engine.ready_queue.get(timeout=10.0)
            print("[Env] Vision Connected!")
        except:
            print("\n[CRITICAL ERROR] Vision Engine timed out!")
            sys.exit(1)

        self.stack_size = 4
        self.attempt_roi = (56, 3, 53, 13)
        self.lower_spectrum = np.array([70, 50, 50])
        self.upper_spectrum = np.array([90, 255, 255])

        self.frame_stack = collections.deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

        self.last_color_mask = None
        self.steps_since_reset = 0

    def reset(self):
        self.controller.act(0)
        self.last_color_mask = None
        self.steps_since_reset = 0
        self.flush_vision()
        return self.get_state()

    def flush_vision(self):
        q_size = self.engine.ready_queue.qsize()
        if q_size > self.stack_size:
            for _ in range(q_size - self.stack_size):
                try:
                    self.engine.ready_queue.get_nowait()
                except:
                    break

    def get_state(self):
        return np.concatenate(list(self.frame_stack), axis=2)

    def step(self, action):
        self.controller.act(action)
        self.steps_since_reset += 1

        if self.engine.ready_queue.qsize() > 2:
            self.flush_vision()

        raw_frame = None
        for _ in range(3):
            try:
                raw_frame, _ = self.engine.ready_queue.get(timeout=0.1)
                break
            except:
                continue

        if raw_frame is None:
            return self.get_state(), 0.0, False, {"warning": "frame_skip"}

        current_frame = raw_frame.copy()
        if hasattr(self.engine, 'idle_queue'):
            self.engine.idle_queue.put(raw_frame)

        tx, ty, tw, th = self.attempt_roi
        text_crop = current_frame[ty:ty + th, tx:tx + tw]
        hsv_crop = cv2.cvtColor(text_crop, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_crop, self.lower_spectrum, self.upper_spectrum)

        is_dead = False
        if self.last_color_mask is not None:
            mask_diff = cv2.bitwise_xor(color_mask, self.last_color_mask)
            if cv2.countNonZero(mask_diff) > 5:
                is_dead = True

        self.last_color_mask = color_mask.copy()
        self.frame_stack.append(current_frame)

        if is_dead:
            reward = -25.0
            done = True
        else:
            reward = 0.1 + (self.steps_since_reset * 0.001)
            done = False

        return self.get_state(), reward, done, {"drops": self.engine.drop_count}


_ = NSApplication.sharedApplication()