import numpy as np
import collections
import Quartz
from AppKit import NSApplication, NSDate, NSRunLoop
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
        else:
            if self.is_holding:
                self._post_event(False)
                self.is_holding = False

    def _post_event(self, key_down):
        event = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, key_down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class GeometryDashEnv:
    def __init__(self):
        self.engine = start_capture()
        self.controller = KeyboardController()
        print("[Env] Vision & Input Connected.")

        self.stack_size = 4
        self.progress_bar_roi = (181, 0, 225, 16)

        self.frame_stack = collections.deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

    def reset(self):
        self.controller.act(0)

        while not self.engine.ready_queue.empty():
            try:
                buf, _ = self.engine.ready_queue.get_nowait()
                self.engine.idle_queue.put(buf)
            except:
                pass

        if len(self.frame_stack) < 4:
            return np.zeros((332, 588, 12), dtype=np.uint8)
        return np.concatenate(self.frame_stack, axis=2)

    def step(self, action):
        self.controller.act(action)

        raw_frame, ts = self.engine.ready_queue.get()

        current_frame = raw_frame.copy()
        self.engine.idle_queue.put(raw_frame)

        x, y, w, h = self.progress_bar_roi
        if y + h <= current_frame.shape[0] and x + w <= current_frame.shape[1]:
            current_frame[y:y + h, x:x + w] = 0

        self.frame_stack.append(current_frame)

        done = False
        reward = 0.1

        state = np.concatenate(self.frame_stack, axis=2)

        return state, reward, done, {}


_ = NSApplication.sharedApplication()