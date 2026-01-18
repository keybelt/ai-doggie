import objc
import cv2
import numpy as np
import time
import Quartz
from Foundation import NSObject, NSRunLoop, NSDate
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCShareableContent, SCContentFilter
from CoreMedia import CMSampleBufferGetImageBuffer, CMSampleBufferIsValid, CMTimeMake


class GDAIVision(NSObject):
    def init(self):
        self = objc.super(GDAIVision, self).init()
        if self is None: return None
        self.context = Quartz.CIContext.contextWithOptions_(None)
        self.frame_times = []
        self.last_frame_time = time.time()
        self.latest_frame = None
        self.new_frame_available = False
        return self

    @objc.typedSelector(b"v@:@@Q")
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, kind):
        if not CMSampleBufferIsValid(sampleBuffer):
            return

        # 1. FPS Tracking
        current_time = time.time()
        self.frame_times.append(current_time - self.last_frame_time)
        if len(self.frame_times) > 120: self.frame_times.pop(0)
        self.last_frame_time = current_time

        # 2. Extract Buffer
        pixel_buffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        ci_image = Quartz.CIImage.imageWithCVPixelBuffer_(pixel_buffer)
        if ci_image is None: return

        extent = ci_image.extent()
        width, height = int(extent.size.width), int(extent.size.height)

        # 3. Create CGImage
        cg_image = self.context.createCGImage_fromRect_(ci_image, extent)
        if not cg_image: return

        # 4. Fast Vectorized Memory Mapping
        bytes_per_row = Quartz.CGImageGetBytesPerRow(cg_image)
        data_provider = Quartz.CGImageGetDataProvider(cg_image)
        pixel_data = Quartz.CGDataProviderCopyData(data_provider)
        raw_bytes = np.frombuffer(pixel_data, dtype=np.uint8)

        actual_rows = len(raw_bytes) // bytes_per_row
        reshaped = raw_bytes[:actual_rows * bytes_per_row].reshape((actual_rows, bytes_per_row))
        clean_rgba = reshaped[:, :width * 4].reshape((actual_rows, width, 4))

        # 5. DYNAMIC CROP
        # If the original window height was ~692, the title bar is ~28px.
        # That is roughly 4% of the screen height.
        # We calculate this dynamically so it works at 90p or 1080p.
        crop_percent = 0.041  # 28 / 692
        title_bar_h = int(height * crop_percent)

        # Slice: remove top title bar
        cropped_rgba = clean_rgba[title_bar_h:, :, :]

        # 6. Final Conversion
        self.latest_frame = cv2.cvtColor(cropped_rgba, cv2.COLOR_RGBA2BGR)
        self.new_frame_available = True


def start_continuous_capture():
    vision = GDAIVision.alloc().init()

    def handler(content, error):
        if error: return
        target = next((w for w in content.windows() if "geometry dash" in (w.title() or "").lower()), None)
        if not target:
            print("Geometry Dash not found.")
            return

        filter_ = SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
        config = SCStreamConfiguration.alloc().init()

        # --- RESOLUTION & SCALING ---
        # Set your desired low-res dimensions
        capture_w, capture_h = 558, 332
        config.setWidth_(capture_w)
        config.setHeight_(capture_h)

        # CRITICAL: Tell macOS to scale the high-res window down to 160x90
        config.setScalesToFit_(True)

        # --- 120 FPS UNLOCK ---
        config.setMinimumFrameInterval_(CMTimeMake(1, 120))
        config.setQueueDepth_(5)
        config.setPixelFormat_(1111970369)  # 'BGRA'
        config.setColorSpaceName_(Quartz.kCGColorSpaceSRGB)

        stream = SCStream.alloc().initWithFilter_configuration_delegate_(filter_, config, vision)
        vision.stream_ref = stream
        stream.addStreamOutput_type_sampleHandlerQueue_error_(vision, 0, None, None)
        stream.startCaptureWithCompletionHandler_(
            lambda err: print(f"120Hz Pipeline @ {capture_w}x{capture_h}") if not err else print(err))

    SCShareableContent.getShareableContentWithCompletionHandler_(handler)
    return vision


if __name__ == "__main__":
    from AppKit import NSApplication

    _ = NSApplication.sharedApplication()
    vision_engine = start_continuous_capture()

    try:
        while True:
            NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.001))

            if vision_engine.new_frame_available:
                avg_fps = 1.0 / (sum(vision_engine.frame_times) / len(
                    vision_engine.frame_times)) if vision_engine.frame_times else 0
                frame = vision_engine.latest_frame

                # Check resolution in real-time
                h, w, _ = frame.shape
                cv2.putText(frame, f"FPS: {int(avg_fps)} RES: {w}x{h}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # Use cv2.WINDOW_NORMAL so you can resize the preview window to see the 160x90 clearly
                cv2.namedWindow("M4 Pro Vision", cv2.WINDOW_NORMAL)
                cv2.imshow("M4 Pro Vision", frame)

                vision_engine.new_frame_available = False
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()