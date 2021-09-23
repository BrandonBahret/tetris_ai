import WindowCamera

from FrameGrabber.FrameSource.FrameSourceBase import FrameSourceBase


class WindowSource(FrameSourceBase):

    def __init__(self, window_title):
        super().__init__()
        self.window_title = window_title
        self.resource = self.load_resource(window_title)

    def get_title(self):
        return self.window_title

    def load_resource(self, window_title):
        window = WindowCamera.find_window(window_title = window_title)
        window_camera = WindowCamera.capture(window, read_from_desktop = True)
        return window_camera

    def read(self):
        success, frame = self.resource.read()
        if success is True:
            frame = frame[32:-2, 4:-4, [0,1,2]]
            frame = frame.copy()
        return (success, frame)
