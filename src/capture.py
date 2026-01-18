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
        self.context = Quartz.CIContext.contextWithOptions_(None)

        # SYNC TOOLS
        self.frame_ready_event = threading.Event()
        self.latest_frame = None

        # Performance Tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.frame_count = 0
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        if not CoreMedia.CMSampleBufferIsValid(sampleBuffer):
            return

        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        if len(self.frame_times) > 120: self.frame_times.pop(0)
        self.last_frame_time = current_time
        self.frame_count += 1

        pixel_buffer = CoreMedia.CMSampleBufferGetImageBuffer(sampleBuffer)
        ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(pixel_buffer)
        if ci_image is None: return

        extent = ci_image.extent()
        width, height = int(extent.size.width), int(extent.size.height)
        cg_image = self.context.createCGImage_fromRect_(ci_image, extent)
        if not cg_image: return

        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        data_provider = Quartz.CGImageGetDataProvider(cg_image)
        pixel_data = Quartz.CGDataProviderCopyData(data_provider)
        raw_bytes = np.frombuffer(pixel_data, dtype=np.uint8)

        clean_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for y in range(height):
            start = y * bytes_per_row
            end = start + (width * 4)
            if end <= len(raw_bytes):
                clean_rgba[y] = raw_bytes[start:end].reshape(width, 4)

        # --- THE GEOMETRIC FIX ---
        # 1. Strip the 28px title bar from the TOP
        # 2. Strip potential black bars from the RIGHT (Slicing width)
        # We target a clean interior area.
        y_start = 28 if height > 300 else 0  # Adjust based on your resolution
        x_end = width - 4 if width > 500 else width  # Snip right edge if black bar exists

        processed_frame = clean_rgba[y_start:, :x_end]

        # Final Resize to ensure the AI always gets exactly the same shape
        # Even if the window moves or resizes slightly.
        self.latest_frame = cv2.resize(
            cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR),
            (588, 332)
        )

        self.frame_ready_event.set()


def start_continuous_capture():
    vision = GDAIVision.alloc().init()

    def handler(content, error):
        if error: return
        target = next((w for w in content.windows() if "geometry dash" in (w.title() or "").lower()), None)
        if not target: return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()

        # We capture slightly larger than needed to ensure we get the full interior
        # Then we crop the bar in the delegate.
        config.setWidth_(target.frame().size.width)
        config.setHeight_(target.frame().size.height)

        config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, 120))
        config.setShowsCursor_(False)
        config.setQueueDepth_(3)
        config.setPixelFormat_(1111970369)  # 'BGRA'

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream

        queue = dispatch_get_global_queue(33, 0)
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, queue, None)
        stream.startCaptureWithCompletionHandler_(
            lambda err: print("\nClean 120Hz Stream Active") if not err else print(err))

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return vision


if __name__ == "__main__":
    from AppKit import NSApplication

    _ = NSApplication.sharedApplication()

    vision_engine = start_continuous_capture()

    try:
        while True:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))

            if vision_engine.frame_ready_event.wait(timeout=0.1):
                vision_engine.frame_ready_event.clear()

                frame = vision_engine.latest_frame

                if vision_engine.frame_count % 30 == 0:
                    avg_fps = 1.0 / (sum(vision_engine.frame_times) / len(
                        vision_engine.frame_times)) if vision_engine.frame_times else 0
                    print(f"\rSynced FPS: {int(avg_fps)} | State: Clean", end="", flush=True)

                cv2.imshow("M4 Pro AI Input (Cropped & Synced)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()