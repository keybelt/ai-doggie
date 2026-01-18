import objc
import cv2
import numpy as np
import time
import Quartz
from Foundation import NSObject, NSRunLoop, NSDate
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCShareableContent, SCContentFilter
from CoreMedia import CMSampleBufferGetImageBuffer, CMSampleBufferIsValid


class GDAIVision(NSObject):
    def init(self):
        self = objc.super(GDAIVision, self).init()
        if self is None: return None
        self.context = Quartz.CIContext.contextWithOptions_(None)

        # Performance Tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.latest_frame = None
        self.new_frame_available = False
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        if not CMSampleBufferIsValid(sampleBuffer):
            return

        # 1. Update FPS Tracking
        current_time = time.time()
        delta = current_time - self.last_frame_time
        self.frame_times.append(delta)
        if len(self.frame_times) > 120: self.frame_times.pop(0)
        self.last_frame_time = current_time

        # 2. Extract Buffer
        pixel_buffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(pixel_buffer)
        if ci_image is None: return

        extent = ci_image.extent()
        width, height = int(extent.size.width), int(extent.size.height)

        # 3. Fast Render
        cg_image = self.context.createCGImage_fromRect_(ci_image, extent)
        if not cg_image: return

        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        data_provider = Quartz.CGImageGetDataProvider(cg_image)
        pixel_data = Quartz.CGDataProviderCopyData(data_provider)
        raw_bytes = np.frombuffer(pixel_data, dtype=np.uint8)

        # 4. Safe Memory Map (Optimized Row-Slicing)
        clean_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            offset = y * bytes_per_row
            row_end = offset + (width * 4)
            if row_end <= len(raw_bytes):
                clean_rgba[y] = raw_bytes[offset:row_end].reshape(width, 4)

        # 5. Crop Title Bar & Convert (BGR for AI/OpenCV)
        y_start = 28 if height == 692 else 0

        # We store the frame in a way the AI can grab it
        self.latest_frame = cv2.cvtColor(clean_rgba[y_start:, :], cv2.COLOR_RGBA2BGR)
        self.new_frame_available = True


def start_continuous_capture():
    vision = GDAIVision.alloc().init()

    def handler(content, error):
        if error: return
        target = next((w for w in content.windows() if "geometry dash" in (w.title() or "").lower()), None)
        if not target: return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()

        # Use target dimensions
        config.setWidth_(target.frame().size.width)
        config.setHeight_(target.frame().size.height)

        # CRITICAL FOR 120 FPS:
        config.setQueueDepth_(3)  # Small queue reduces latency
        config.setPixelFormat_(1111970369)  # 'BGRA'

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, None, None)
        stream.startCaptureWithCompletionHandler_(lambda err: print("120Hz Pipeline Online") if not err else print(err))

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return vision


if __name__ == "__main__":
    from AppKit import NSApplication

    _ = NSApplication.sharedApplication()

    vision_engine = start_continuous_capture()

    print("Press 'q' to stop.")
    try:
        while True:
            # Pumping the event loop at high frequency
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))

            if vision_engine.new_frame_available:
                # Calculate current average FPS
                avg_fps = 1.0 / (sum(vision_engine.frame_times) / len(
                    vision_engine.frame_times)) if vision_engine.frame_times else 0

                # Show frame with FPS overlay
                frame = vision_engine.latest_frame
                cv2.putText(frame, f"FPS: {int(avg_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("M4 Pro High-Speed Vision", frame)
                vision_engine.new_frame_available = False  # Reset flag

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()