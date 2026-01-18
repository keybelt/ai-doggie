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

        self.BUFFER_SIZE = 60
        self.idle_queue = queue.Queue()
        self.ready_queue = queue.Queue()

        for _ in range(self.BUFFER_SIZE):
            buf = np.zeros((332, 588, 3), dtype=np.uint8)
            self.idle_queue.put(buf)

        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        try:
            frame_buffer = self.idle_queue.get_nowait()
        except queue.Empty:
            return

        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        if not pixel_buffer:
            self.idle_queue.put(frame_buffer)
            return

        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 1)
        try:
            width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
            height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
            base_addr = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)

            raw_buffer = base_addr.as_buffer(bpr * height)
            raw_array = np.frombuffer(raw_buffer, dtype=np.uint8).reshape(height, bpr)

            np.copyto(frame_buffer, raw_array[:, :width * 4].reshape(height, width, 4)[:, :, :3])

        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 1)

        self.ready_queue.put((frame_buffer, time.perf_counter()))


def start_capture():
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
        config.setQueueDepth_(6)
        config.setPixelFormat_(1111970369)

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream

        q = dispatch_queue_create(b"com.gdai.capture", None)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, q, None)
        stream.startCaptureWithCompletionHandler_(lambda err: None)

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return vision