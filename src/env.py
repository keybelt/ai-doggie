import queue

import numpy as np
import sys
import Quartz
import atexit
from AppKit import NSApplication
from capture import start_capture


class KeyboardController:
    def __init__(self):
        self.is_holding = False
        self.SPACE_KEY = 49
        self.src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        atexit.register(self.cleanup)

    def cleanup(self):
        if self.is_holding:
            ev_up = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, False)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
            self.is_holding = False

    def act(self, action):
        if action == 1:
            if not self.is_holding:
                ev_down = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, True)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_down)
                self.is_holding = True
                return True
        else:
            if self.is_holding:
                ev_up = Quartz.CGEventCreateKeyboardEvent(self.src, self.SPACE_KEY, False)
                Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev_up)
                self.is_holding = False
        return False


class GeometryDashEnv:
    def __init__(self):
        self.engine = start_capture()
        self.controller = KeyboardController()
        self.last_valid_frame = np.zeros((332, 588, 3), dtype=np.uint8)

        print("[Env] Connecting to Vision Engine...")
        self.engine.paused = False

        try:
            raw_frame, _ = self.engine.ready_queue.get(timeout=10.0)
            self.engine.idle_queue.put(raw_frame)
            print("[Env] Vision Connected!")
        except:
            print("\n[CRITICAL ERROR] Vision Engine timed out!")
            sys.exit(1)

        self.flush_vision()

    def reset(self):
        self.flush_vision()
        self.controller.cleanup()

    def flush_vision(self):
        if self.engine.ready_queue.empty():
            return 0

        skipped_in_batch = 0
        while self.engine.ready_queue.qsize() > 1:
            try:
                frame, _ = self.engine.ready_queue.get_nowait()
                self.engine.idle_queue.put(frame)
                skipped_in_batch += 1
            except:
                break

        return skipped_in_batch

    def get_frame(self):
        flushed_count = self.flush_vision()

        try:
            raw_frame, timestamp = self.engine.ready_queue.get(timeout=0.1)

            current_frame = raw_frame[:, :, 2::-1].copy()

            self.engine.idle_queue.put_nowait(raw_frame)
            self.last_valid_frame = current_frame

            return current_frame, timestamp, flushed_count
        except queue.Empty:
            return self.last_valid_frame, 0.0, flushed_count

    def step(self, action):
        self.controller.act(action)
        return self.get_frame()


_ = NSApplication.sharedApplication()