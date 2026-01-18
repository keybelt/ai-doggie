import objc
import numpy as np
import time
import Quartz
import CoreMedia
import queue
from Foundation import NSObject, NSRunLoop, NSDate
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCShareableContent, SCContentFilter
from libdispatch import dispatch_queue_create


class GDAIVision(NSObject):
    def init(self):
        self = objc.super(GDAIVision, self).init()
        if self is None: return None

        # --- CONFIGURATION ---
        # 60 Frames = 0.5 seconds of buffer at 120Hz.
        # This absorbs massive lag spikes without dropping a single frame.
        self.BUFFER_SIZE = 60

        self.idle_queue = queue.Queue()
        self.ready_queue = queue.Queue()
        self.true_drop_count = 0  # ACTUAL data loss counter

        # Pre-allocate ALL memory upfront. Zero allocations during runtime.
        for _ in range(self.BUFFER_SIZE):
            buf = np.zeros((332, 588, 3), dtype=np.uint8)
            self.idle_queue.put(buf)

        self.processing_latency_ms = 0.0
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        # 1. HARDWARE INTERRUPT (Run as fast as possible)
        # Check if we have a free buffer to write into
        try:
            frame_buffer = self.idle_queue.get_nowait()
        except queue.Empty:
            # CRITICAL: This is the ONLY time a real drop happens.
            # It means Main Thread fell 0.5 seconds behind.
            self.true_drop_count += 1
            return

        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        if not pixel_buffer:
            self.idle_queue.put(frame_buffer)  # Return unused buffer
            return

        # 2. ZERO-COPY MAP & FAST COPY
        # We use the raw address to avoid ObjC overhead
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 1)
        try:
            width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
            height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
            address = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)

            # Create a numpy view directly on the IOSurface (Hardware Memory)
            raw_buffer = address.as_buffer(bpr * height)
            raw_array = np.frombuffer(raw_buffer, dtype=np.uint8).reshape(height, bpr)

            # Fast memcpy to our private buffer
            np.copyto(frame_buffer, raw_array[:, :width * 4].reshape(height, width, 4)[:, :, :3])

        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 1)

        # 3. PUSH TO CONSUMER
        self.ready_queue.put((frame_buffer, time.perf_counter()))


def start_continuous_capture():
    vision = GDAIVision.alloc().init()

    def handler(content, error):
        if error: return
        target = next((w for w in content.windows() if "geometry dash" in (w.title() or "").lower()), None)
        if not target: return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()
        config.setSourceRect_(Quartz.CGRectMake(0, 28, 1176, 664))
        config.setWidth_(588)
        config.setHeight_(332)
        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 120))
        config.setShowsCursor_(False)
        config.setQueueDepth_(10)
        config.setPixelFormat_(1111970369)  # BGRA

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream

        # Serial Queue ensures frames arrive in order 1, 2, 3...
        queue_gcd = dispatch_queue_create(b"com.gdai.capture", None)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, queue_gcd, None)
        stream.startCaptureWithCompletionHandler_(lambda err: print("\n[M4 Pro] High-Performance Engine Started"))

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return vision


if __name__ == "__main__":
    from AppKit import NSApplication

    _ = NSApplication.sharedApplication()
    engine = start_continuous_capture()

    total_processed = 0

    try:
        while True:
            try:
                # 1. ATTEMPT TO GET FRAME (Non-Blocking)
                # We do not wait here because we want to manage the RunLoop manually
                frame_data = engine.ready_queue.get_nowait()

                # --- [HOT PATH] WE HAVE DATA ---
                current_frame, produced_ts = frame_data

                # ... YOUR LOGIC HERE ...

                # Metrics
                latency = (time.perf_counter() - produced_ts) * 1000
                total_processed += 1

                # Recycle Buffer Immediately
                engine.idle_queue.put(current_frame)

                if total_processed % 120 == 0:
                    # Queue Depth = How many frames are waiting
                    q_depth = engine.ready_queue.qsize()
                    print(f"\rLatency: {latency:.2f}ms | Queue Depth: {q_depth} | TRUE DROPS: {engine.true_drop_count}",
                          end="")

            except queue.Empty:
                # --- [COLD PATH] NO DATA ---
                # Only pump the OS loop when we are waiting.
                # This prevents the OS loop from stealing cycles during active processing.
                NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))

    except KeyboardInterrupt:
        pass