import numpy as np
import collections
import cv2
import time
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
        else:
            if self.is_holding:
                self._post_event(False)
                self.is_holding = False

    def _post_event(self, key_down):
        event = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, key_down)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


class GeometryDashEnv:
    def __init__(self):
        # 1. Start Capture
        self.engine = start_capture()
        self.controller = KeyboardController()

        # 2. [FIX] Connection Handshake
        # We wait here to confirm the Capture Engine is actually working.
        # If we don't get a frame in 5 seconds, we abort with a helpful error.
        print("[Env] Connecting to Vision Engine...")
        try:
            # Try to get 1 frame with a 5-second timeout
            self.engine.ready_queue.get(timeout=5.0)
            print("[Env] Vision Connected! Stream is 120Hz live.")
        except:
            print("\n[CRITICAL ERROR] Vision Engine timed out!")
            print("Possible causes:")
            print("1. Geometry Dash is not open.")
            print("2. The window title doesn't match 'Geometry Dash'.")
            print("3. Screen Recording permissions are denied.")
            sys.exit(1)

        self.stack_size = 4

        # --- CONFIG ---
        self.progress_bar_roi = (184, 0, 221, 15)
        self.bar_threshold = 150

        self.frame_stack = collections.deque(maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.frame_stack.append(np.zeros((332, 588, 3), dtype=np.uint8))

        self.last_fill_count = 0

    def reset(self):
        self.controller.act(0)
        self.last_fill_count = 0

        # Flush queue
        while not self.engine.ready_queue.empty():
            try:
                self.engine.ready_queue.get_nowait()
            except:
                pass

        if len(self.frame_stack) < 4:
            return np.zeros((332, 588, 12), dtype=np.uint8)
        return np.concatenate(self.frame_stack, axis=2)

    Python

    def step(self, action):
        self.controller.act(action)

        # --- [NEW] LAG BUSTER ---
        # If queue has > 1 frame, we are behind. Drop everything to catch up.
        if self.engine.ready_queue.qsize() > 1:
            dropped = 0
            while not self.engine.ready_queue.empty():
                try:
                    trash, _ = self.engine.ready_queue.get_nowait()
                    self.engine.idle_queue.put(trash)  # Recycle memory
                    dropped += 1
                except:
                    break
            # print(f"⚠️ Recovered from lag! Skipped {dropped} frames.")

        # Now grab the freshest frame (with a safety timeout)
        try:
            raw_frame, ts = self.engine.ready_queue.get(timeout=1.0)
        except:
            # Prevent crash if game is closed
            raw_frame = np.zeros((332, 588, 3), dtype=np.uint8)

        current_frame = raw_frame.copy()

        # Recycle buffer only if it's a real shared memory frame
        if hasattr(self.engine, 'idle_queue'):
            self.engine.idle_queue.put(raw_frame)

        # --- 1. FILTER PROGRESS BAR ---
        x, y, w, h = self.progress_bar_roi
        bar_crop = current_frame[y:y + h, x:x + w]

        gray = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.bar_threshold, 255, cv2.THRESH_BINARY)

        # --- 2. COUNT PIXELS ---
        current_fill = cv2.countNonZero(binary)
        delta = self.last_fill_count - current_fill

        # --- 3. DEATH LOGIC ---
        is_dead = False

        # [FIX] Increased Threshold
        # '1' is too unsafe. Render noise or compression can flicker 1-5 pixels.
        # A death resets the WHOLE bar (usually 100+ pixels).
        # 50 is a safe number that won't trigger false positives but catches every reset.
        if delta > 10:
            is_dead = True

        self.last_fill_count = current_fill

        # --- 4. BLINDFOLD ---
        if y + h <= current_frame.shape[0] and x + w <= current_frame.shape[1]:
            current_frame[y:y + h, x:x + w] = 0

        self.frame_stack.append(current_frame)

        if is_dead:
            done = True
            reward = -10.0
            self.last_fill_count = 0
        else:
            done = False
            reward = 0.1

        state = np.concatenate(self.frame_stack, axis=2)
        return state, reward, done, {}


_ = NSApplication.sharedApplication()