import objc
import numpy as np
import cv2
import sys
import queue
import Quartz
import CoreMedia
from Foundation import NSObject, NSRunLoop, NSDate
from AppKit import NSApplication
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCShareableContent, SCContentFilter
from libdispatch import dispatch_queue_create

# --- CENTRAL CONFIGURATION ---
CONFIG = {
    "WIDTH": 558,  # Target Width
    "HEIGHT": 332,  # Target Height
    "WINDOW_NAME": "geometry dash",
    "BUFFER_SIZE": 3,
    "OUTPUT_NAME": "frame.png"
}

# Derived value: 4 bytes per pixel for BGRA
CONFIG["BYTES_PER_ROW_LIMIT"] = CONFIG["WIDTH"] * 4


class GDAIVision(NSObject):
    def init(self):
        self = objc.super(GDAIVision, self).init()
        if self is None: return None

        self.idle_queue = queue.Queue()
        self.ready_queue = queue.Queue()
        self.stream_ref = None

        # Initialize buffers using centralized CONFIG
        for _ in range(CONFIG["BUFFER_SIZE"]):
            buf = np.zeros((CONFIG["HEIGHT"], CONFIG["WIDTH"], 3), dtype=np.uint8)
            self.idle_queue.put(buf)
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        if not pixel_buffer: return

        try:
            frame_buffer = self.idle_queue.get_nowait()
        except queue.Empty:
            return

        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 1)
        try:
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
            base_addr = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)

            # Use CONFIG for height and slicing
            raw_buffer = base_addr.as_buffer(bpr * CONFIG["HEIGHT"])
            raw_array = np.frombuffer(raw_buffer, dtype=np.uint8).reshape(CONFIG["HEIGHT"], bpr)

            # Slice the valid pixel data and convert BGRA -> RGB
            np.copyto(frame_buffer,
                      raw_array[:, :CONFIG["BYTES_PER_ROW_LIMIT"]]
                      .reshape(CONFIG["HEIGHT"], CONFIG["WIDTH"], 4)[:, :, [2, 1, 0]])
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 1)

        self.ready_queue.put(frame_buffer)


def start_capture(vision):
    def handler(content, error):
        if error:
            print(f"Error: {error}")
            return

        target = next((w for w in content.windows() if CONFIG["WINDOW_NAME"] in (w.title() or "").lower()), None)
        if not target:
            print(f"Window '{CONFIG['WINDOW_NAME']}' not found!")
            return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()

        # Apply centralized resolution
        config.setWidth_(CONFIG["WIDTH"])
        config.setHeight_(CONFIG["HEIGHT"])
        config.setPixelFormat_(1111970369)  # BGRA

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream
        q = dispatch_queue_create(b"com.gdai.capture", None)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, q, None)
        stream.startCaptureWithCompletionHandler_(lambda err: print("Stream active...") if not err else print(err))

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)


if __name__ == "__main__":
    # Ensure macOS App Context
    app = NSApplication.sharedApplication()

    vision_instance = GDAIVision.alloc().init()
    start_capture(vision_instance)

    print(f"Capturing a {CONFIG['WIDTH']}x{CONFIG['HEIGHT']} frame...")

    captured = False
    while not captured:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

        try:
            frame = vision_instance.ready_queue.get_nowait()

            # Save frame (RGB to BGR for OpenCV)
            cv2.imwrite(CONFIG["OUTPUT_NAME"], cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            print(f"Saved: {CONFIG['OUTPUT_NAME']}")

            if vision_instance.stream_ref:
                vision_instance.stream_ref.stopCaptureWithCompletionHandler_(None)

            captured = True
        except queue.Empty:
            continue

    print("Task complete.")
    sys.exit(0)