import numpy as np
import cv2

import d3dshot
import WindowCamera

from FrameGrabber.FrameSource.FrameSourceBase import FrameSourceBase


class D3DSource(FrameSourceBase):
    def __init__(self, window_title):
        super().__init__()
        self.window_title = window_title
        self.window = WindowCamera.find_window(window_title = window_title)
        self.resource = self.load_resource(window_title)

    def get_title(self):
        return self.window_title

    def get_region(self):
        left = self.window.x
        top = self.window.y
        right = left + self.window.width
        bottom = top + self.window.height
        return (left+2, top+32, right-2, bottom+32)

    def load_resource(self, window_title):
        d = d3dshot.create(capture_output="numpy")
        return d
        
    def read(self):
        frame = self.resource.screenshot(region=self.get_region())
        success = (frame is not None)
        if success:
            frame = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            success = True
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
        return (success, frame)
