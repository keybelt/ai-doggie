import threading
import ctypes
import numpy as np
import objc
from Foundation import NSObject
from ScreenCaptureKit import SCStreamOutputTypeScreen
from Quartz import (
    CVPixelBufferLockBaseAddress, CVPixelBufferUnlockBaseAddress,
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetHeight, CVPixelBufferGetWidth,
    kCVPixelBufferLock_ReadOnly
)
from CoreMedia import CMSampleBufferGetImageBuffer, CMSampleBufferIsValid


class CaptureDelegate(NSObject):
    def init(self):
        self = objc.super(CaptureDelegate, self).init()
        self.frame = None
        self.event = threading.Event()

        return self

    @objc.signature(b"v@:@@Q")
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