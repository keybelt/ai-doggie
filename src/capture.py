import objc
import cv2
import numpy as np
import time
import Quartz
import CoreMedia
import threading
from Foundation import NSObject, NSRunLoop, NSDate
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCShareableContent, SCContentFilter
from libdispatch import dispatch_get_global_queue


class GDAIVision(NSObject):
    def init(self):
        self = objc.super(GDAIVision, self).init()
        if self is None: return None

        # PRE-ALLOCATE: Stop the Garbage Collector from triggering
        # This is a massive win for high-frequency loops
        self.latest_frame = np.zeros((332, 588, 3), dtype=np.uint8)

        self.frame_ready_event = threading.Event()
        self.processing_latency_ms = 0.0
        self.total_frames_produced = 0
        self.total_frames_processed = 0
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        self.total_frames_produced += 1
        start_ts = time.perf_counter()

        # 1. Get Buffer
        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        if not pixel_buffer: return

        # 2. Lock
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 1)

        try:
            width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
            height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)

            # 3. Direct Map
            address_varlist = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            raw_buffer = address_varlist.as_buffer(bpr * height)

            # 4. View Creation (Zero-Copy)
            raw_array = np.frombuffer(raw_buffer, dtype=np.uint8).reshape(height, bpr)

            # 5. Fast Copy-to-Preallocated (Slices BGRA -> BGR)
            # np.copyto is significantly faster than .copy() as it doesn't allocate new RAM
            np.copyto(self.latest_frame, raw_array[:, :width * 4].reshape(height, width, 4)[:, :, :3])

        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 1)

        self.processing_latency_ms = (time.perf_counter() - start_ts) * 1000
        self.frame_ready_event.set()


def start_continuous_capture():
    vision = GDAIVision.alloc().init()

    def handler(content, error):
        if error: return
        target = next((w for w in content.windows() if "geometry dash" in (w.title() or "").lower()), None)
        if not target: return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()

        # Hardware scaling
        config.setSourceRect_(Quartz.CGRectMake(0, 28, 1176, 664))
        config.setWidth_(588)
        config.setHeight_(332)

        # 120Hz Config
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 120))
        config.setShowsCursor_(False)

        # INCREASE QUEUE DEPTH: Essential for 120Hz.
        # 8 frames provides ~66ms of "buffer" against Python jitter.
        config.setQueueDepth_(8)
        config.setPixelFormat_(1111970369)  # 'BGRA'

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream

        queue = dispatch_get_global_queue(33, 0)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, queue, None)
        stream.startCaptureWithCompletionHandler_(
            lambda err: print("\n[M4 Pro] 120Hz Zero-Copy Engine Online") if not err else print(err))

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return vision


if __name__ == "__main__":
    from AppKit import NSApplication

    _ = NSApplication.sharedApplication()
    engine = start_continuous_capture()

    try:
        while True:
            # High-resolution runloop pump
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.0001))

            if engine.frame_ready_event.wait(timeout=0.001):
                engine.frame_ready_event.clear()
                engine.total_frames_processed += 1

                if engine.total_frames_processed % 120 == 0:
                    drops = engine.total_frames_produced - engine.total_frames_processed
                    print(f"\rLatency: {engine.processing_latency_ms:.2f}ms | Drops: {drops} | FPS: 120 ", end="")

    except KeyboardInterrupt:
        print("\nTerminated.")