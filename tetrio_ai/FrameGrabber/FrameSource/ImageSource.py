import cv2

from FrameGrabber.FrameSource.FrameSourceBase import FrameSourceBase

class ImageSource(FrameSourceBase):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.resource = self.load_resource(filename)

    def get_title(self):
        return self.filename

    def load_resource(self, filename):
        image_file = cv2.imread(filename)
        return image_file

    def read(self):
        return (True, self.resource)
