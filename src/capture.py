import objc
import time
import cv2
import numpy as np
import os
import Quartz
from Foundation import NSObject, NSRunLoop, NSDate
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCShareableContent, SCContentFilter
from CoreMedia import CMSampleBufferGetImageBuffer, CMSampleBufferIsValid

class SingleFrameDelegate(NSObject):
    def init(self):
        self = objc.super(SingleFrameDelegate, self).init()
        self.context = Quartz.CIContext.contextWithOptions_(None)
        self.captured = False
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        if self.captured or not CMSampleBufferIsValid(sampleBuffer):
            return

        pixel_buffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(pixel_buffer)

        if ci_image is None:
            return

        # Get window dimensions
        extent = ci_image.extent()
        width = int(extent.size.width)
        height = int(extent.size.height)

        cg_image = self.context.createCGImage_fromRect_(ci_image, extent)
        if not cg_image:
            return

        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        data_provider = Quartz.CGImageGetDataProvider(cg_image)
        pixel_data = Quartz.CGDataProviderCopyData(data_provider)
        raw_bytes = np.frombuffer(pixel_data, dtype=np.uint8)

        # 1. Reconstruct clean frame (This avoids the reshape crash)
        clean_frame = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            start = y * bytes_per_row
            end = start + (width * 4)
            clean_frame[y] = raw_bytes[start:end].reshape(width, 4)

        if not np.any(clean_frame):
            return

        # 2. Crop Title Bar (28px) if windowed
        if height == 692:
            clean_frame = clean_frame[28:, :]

        # 3. THE COLOR SHIFT LINE
        # This is the line that produces the "Red is Yellow" or "Yellow is Blue" result.
        # ScreenCaptureKit usually outputs BGRA, and OpenCV imwrite expects BGR.
        bgr_frame = cv2.cvtColor(clean_frame, cv2.COLOR_RGBA2BGR)

        # 4. Save
        path = os.path.join(os.getcwd(), "gd_test_frame.png")
        cv2.imwrite(path, bgr_frame)

        print(f"\n>>> SUCCESS: Captured {bgr_frame.shape[1]}x{bgr_frame.shape[0]}")
        self.captured = True

def capture_one_frame():
    delegate = SingleFrameDelegate.alloc().init()

    def handler(content, error):
        if error:
            print(f"Content Error: {error}")
            return

        target = next((w for w in content.windows() if "geometry dash" in (w.title() or "").lower()), None)
        if not target:
            print("!!! ERROR: Geometry Dash window not found.")
            return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()
        config.setWidth_(target.frame().size.width)
        config.setHeight_(target.frame().size.height)
        config.setQueueDepth_(5)
        config.setPixelFormat_(1111970369)  # 'BGRA'

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, delegate)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(delegate, 0, None, None)

        def start_handler(err):
            if err: print(f"Stream Start Error: {err}")
            else: print("Stream is LIVE. Waiting for pixels...")

        stream.startCaptureWithCompletionHandler_(start_handler)

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return delegate

if __name__ == "__main__":
    from AppKit import NSApplication
    _ = NSApplication.sharedApplication()
    active_delegate = capture_one_frame()

    timeout = 10
    start_wait = time.time()
    while not active_delegate.captured and (time.time() - start_wait) < timeout:
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.1))

    if not active_delegate.captured:
        print("\n!!! TIMEOUT: No frame received.")