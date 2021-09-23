import cv2

from FrameGrabber.FrameSource.FrameSourceBase import FrameSourceBase


class VideoSource(FrameSourceBase):

    def __init__(self, filename, starting_frame=1):
        super().__init__()
        self.starting_frame = starting_frame

        self.filename = filename
        self.video_file = self.load_resource(filename)

        # for _ in range(0, self.starting_frame):
        #     self.video_file.read()
        self.video_file.set(cv2.CAP_PROP_POS_FRAMES, self.starting_frame)

    def reset(self):
        self.video_file.set(cv2.CAP_PROP_POS_FRAMES, self.starting_frame)
        # self.video_file.release()
        # self.video_file = self.load_resource(self.filename)
        # for _ in range(0, self.starting_frame):
        #     self.video_file.read()

    def get_title(self):
        return self.filename

    def load_resource(self, filename):
        video_file = cv2.VideoCapture(filename)
        return video_file

    def read(self):
        success, frame = self.video_file.read()
        if success is False:
            self.video_file.release()
            self.video_file = self.load_resource(self.filename)
            success, frame = self.video_file.read()

        # if success is True:
        #     frame = frame[32:-2, 4:-4, [0,1,2]]

        return (success, frame)
