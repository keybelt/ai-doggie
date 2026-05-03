"""Contains the entire screen capture mechanism and is the single source of truth for the frame drop metric.

Examples:
    >>> capture_engine = start_capture_engine()

    Get a frame from the queue:
    >>> capture_engine.queue_full.get()

    Get the frame drop count:
    >>> frame_drops = capture_engine.frame_drops
"""

import queue
from typing import Self, override

import CoreMedia
import numpy as np
import objc
import Quartz
import ScreenCaptureKit as Sck
from Foundation import NSObject
from libdispatch import dispatch_queue_create

from config import CONFIG as _CONFIG
from type_defs import Frame, FrameQueue

_CONFIG_CAPTURE = _CONFIG["capture"]

_PIPELINE_FRAME_WIDTH_PX = _CONFIG_CAPTURE["frameDims"]["pipelineWidthPx"]
_PIPELINE_FRAME_HEIGHT_PX = _CONFIG_CAPTURE["frameDims"]["pipelineHeightPx"]

_QUEUE_DEPTH: int = _CONFIG_CAPTURE["queueDepth"]


class _CaptureEngine(NSObject):
    """Captures frames from ScreenCaptureKit with a push-based system (delegate). Uses a dual-queue mechanism and tracks frame drops."""

    @override
    def init(self) -> Self:
        """Initialize and populate queues, frame drop counter, frame specifications, and capture stream reference."""
        self = objc.super(_CaptureEngine, self).init()

        self.queue_empty: FrameQueue = queue.Queue()
        self.queue_full: FrameQueue = queue.Queue()

        self.frame_drops = 0

        self.capture_stream = None

        for _ in range(_QUEUE_DEPTH):
            color_channel_depth = 4
            frame = np.zeros(
                (
                    _PIPELINE_FRAME_HEIGHT_PX,
                    _PIPELINE_FRAME_WIDTH_PX,
                    color_channel_depth,
                ),
                dtype=np.uint8,
            )
            self.queue_empty.put(frame)

        return self

    # Use type encoding (v@:@@q) to tell macOS what to expect.
    @objc.typedSelector(b"v@:@@q")
    @override
    def stream_didOutputSampleBuffer_ofType_(self, _stream, sample_buffer, _media_type):
        """Use a delegate callback to get a frame immediately as macOS produces one."""
        # Ensure the Obj-C objects get cleaned up properly.
        with objc.autorelease_pool():
            # Extract pixels from frame metadata
            frame_px: CoreMedia.CVImageBuffer = CoreMedia.CMSampleBufferGetImageBuffer(
                sample_buffer,
            )

            try:
                frame_buf: Frame = self.queue_empty.get_nowait()
            except queue.Empty:
                frame_buf: Frame = self.queue_full.get_nowait()
                self.frame_drops += 1

            # Safe access and override prevention.
            read_only_flag = 1
            Quartz.CVPixelBufferLockBaseAddress(frame_px, read_only_flag)

            try:
                # bytes per row contains the padded row length.
                bytes_per_row = Quartz.CVPixelBufferGetBytesPerRow(frame_px)

                frame_ptr = Quartz.CVPixelBufferGetBaseAddress(frame_px)

                # Convert to a 1d memory buffer of size bytes per row * height for the full frame.
                frame_bytes: memoryview = frame_ptr.as_buffer(
                    bytes_per_row * _PIPELINE_FRAME_HEIGHT_PX,
                )

                # Convert to a 2d array to represent width and height.
                frame_arr = np.frombuffer(
                    frame_bytes,
                    dtype=np.uint8,
                ).reshape(_PIPELINE_FRAME_HEIGHT_PX, bytes_per_row)

                # Make sure the buffer fits the frame (color and width axes are combined).
                frame_buf_view = frame_buf.view(
                    np.uint8,
                ).reshape(_PIPELINE_FRAME_HEIGHT_PX, _PIPELINE_FRAME_WIDTH_PX * 4)

                # Crop out the padding.
                np.copyto(frame_buf_view, frame_arr[:, : _PIPELINE_FRAME_WIDTH_PX * 4])
            finally:
                Quartz.CVPixelBufferUnlockBaseAddress(frame_px, read_only_flag)

            self.queue_full.put(frame_buf)

    def stop_capture_stream(self):
        if self.capture_stream:
            self.capture_stream.stopCaptureWithCompletionHandler_()
            self.capture_stream = None


def start_capture_engine() -> _CaptureEngine:
    """Initialize and configure capture window. Set up capture stream and engine and begin capture process."""
    # Initialize with .alloc().init() because it's an inheritance of NSObject.
    capture_engine: _CaptureEngine = _CaptureEngine.alloc().init()

    def on_shareable_content(content: Sck.SCShareableContent) -> None:
        """Asynchronous function that handles the window configurations."""
        src_height_px = _CONFIG_CAPTURE["frameDims"]["srcHeightPx"]
        src_width_px = _CONFIG_CAPTURE["frameDims"]["srcWidthPx"]

        title_bar_crop_px = 28
        pixel_format_bgra: int = 1111970369
        capture_fps = _CONFIG_CAPTURE["fps"]

        try:
            window_target = next(
                w
                for w in content.windows()
                if "geometry dash" in (w.title() or "").lower()
            )

            print("Geometry Dash window found.")

            window_filter = (
                Sck.SCContentFilter.alloc().initWithDesktopIndependentWindow_(
                    window_target,
                )
            )

            config: Sck.SCStreamConfiguration = Sck.SCStreamConfiguration.alloc().init()
            config.setSourceRect_(
                Quartz.CGRectMake(0, title_bar_crop_px, src_width_px, src_height_px),
            )
            config.setWidth_(_PIPELINE_FRAME_WIDTH_PX)
            config.setHeight_(_PIPELINE_FRAME_HEIGHT_PX)
            config.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, capture_fps))
            config.setShowsCursor_(False)
            config.setQueueDepth_(_QUEUE_DEPTH)
            config.setPixelFormat_(pixel_format_bgra)

            capture_stream = (
                Sck.SCStream.alloc().initWithFilter_configuration_delegate_(
                    window_filter,
                    config,
                    capture_engine,
                )
            )
            capture_engine.capture_stream = capture_stream

            print("Capture stream live.")

            # Create a background thread so the capture stream doesn't interrupt the main pipeline.
            dispatch_queue = dispatch_queue_create(b"com.ai-doggie.capture")

            # Tell the capture stream to send frames to the capture engine using type video.
            capture_stream.addStreamOutput_type_sampleHandlerQueue_error_(
                capture_engine,
                0,
                dispatch_queue,
                None,
            )
            capture_stream.startCaptureWithCompletionHandler_()
        except StopIteration:
            err_msg = "Geometry Dash window not found."
            raise Exception(err_msg) from StopIteration

    # Find available windows to begin capture.
    Sck.SCShareableContent.getShareableContentWithCompletionHandler_(
        on_shareable_content,
    )
    return capture_engine
