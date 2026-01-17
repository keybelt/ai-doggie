import threading
import ctypes
import numpy as np
import time
import objc

from Foundation import NSObject, NSRunLoop

from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCContentFilter, SCShareableContent, SCStreamOutputTypeScreen

from Quartz import CVPixelBufferLockBaseAddress, CVPixelBufferUnlockBaseAddress, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight, CVPixelBufferGetWidth, kCVPixelBufferLock_ReadOnly, kCVPixelFormatType_32BGRA

from CoreMedia import CMSampleBufferGetImageBuffer, CMSampleBufferIsValid, CMTimeMake


class CaptureDelegate(NSObject):
    def init(self):
        self = objc.super(CaptureDelegate, self).init()
        self.frame = None
        self.event = threading.Event()

        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        if kind != SCStreamOutputTypeScreen or not CMSampleBufferIsValid(sampleBuffer):
            return

        pixel_buffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly)

        try:
            width = CVPixelBufferGetWidth(pixel_buffer)
            height = CVPixelBufferGetHeight(pixel_buffer)
            row_bytes = CVPixelBufferGetBytesPerRow(pixel_buffer)
            base_address = CVPixelBufferGetBaseAddress(pixel_buffer)

            ctypes_array = (ctypes.c_uint8 * (height * row_bytes)).from_address(int(base_address))
            image = np.ctypeslib.as_array(ctypes_array).reshape(height, row_bytes)
            self.frame = image[:, :width*4].reshape(height, width, 4)[:, :, :3].copy()

            self.event.set()

        finally:
            CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly)


class ScreenCapture:
    def __init__(self):
        self.window_name = "Geometry Dash"
        self.delegate = CaptureDelegate.alloc().init()
        self.width = 588
        self.height = 332
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def get_latest_frame(self):
        self.delegate.event.wait()
        self.delegate.event.clear()

        return self.delegate.frame

    def _run_loop(self):
        def content_callback(content, error):
            if error:
                return

            target = None
            for window in content.windows():
                if self.window_name.lower() in (window.title() or "").lower():
                    target = window
                    break

            if target:
                filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
            else:
                filter_ = SCContentFilter.alloc().initWithDisplay_excludingWindows_(content.displays()[0], [])

            config = SCStreamConfiguration.alloc().init()
            config.setWidth_(self.width)
            config.setHeight_(self.height)
            config.setMinimumFrameInterval_(CMTimeMake(1, 120))
            config.setPixelFormat_(kCVPixelFormatType_32BGRA)
            config.setQueueDepth_(3)

            stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, self.delegate)

        SCShareableContent.getShareableContentWithCompletionHandler_(content_callback)
        NSRunLoop.currentRunLoop().run()


if __name__ == "__main__":
    from AppKit import NSApplication
    app = NSApplication.sharedApplication()

    cap = ScreenCapture()
    cap.start()

    print(1)
    try:
        print(2)
        frame_count = 0
        start_time = time.time()

        while frame_count < 1000:
            print(3)
            frame = cap.get_latest_frame()

            frame_count += 1
            if frame_count % 120 == 0:
                print(f"FPS: {frame_count / (time.time() - start_time):.2f}")

    except KeyboardInterrupt:
        print("Stopped.")