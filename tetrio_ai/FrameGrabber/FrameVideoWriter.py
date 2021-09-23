import numpy as np
import cv2
from threading import Event, Thread
import time

from Helpers.Draw import Draw

from FrameGrabber.FrameSource.FrameSourceBase import FrameSourceBase
from FrameGrabber.Frame import Frame


class FrameVideoWriter:

    def __init__(self, frame_source:FrameSourceBase, target_fps=60, display_frames=False, should_draw_fps=False):
        self.display_frames = display_frames
        self.should_draw_fps = should_draw_fps
        self.frame_source = frame_source
        self.target_fps = target_fps
        
        self.frame_source_is_working = True
        self.current_frame = None
        self.fps = 60

        self.thread_manager = None

        init_frame = self.frame_source.read()
        self.window_size = (init_frame.shape[1], init_frame.shape[0])
        self.created_datetime = time.strftime("%Y-%m-%d %H-%M-%S")
        self.filename = "{0} {1}.avi".format(self.frame_source.get_title(), self.created_datetime)
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.video_file = cv2.VideoWriter(self.filename, self.fourcc, self.target_fps, self.window_size)

        self._start()

    def on_loop(self):
        success, frame = self.frame_source.read()
        self.frame_source_is_working = success
        self.current_frame = frame

        if success:
            self.video_file.write(frame)

    def _start(self):
        self.thread_manager = Event()
        self.thread_manager.set()
        self.on_loop()
        Thread(target=self.main_loop).start()

    def stop(self):
        if self.thread_manager is not None:
            self.thread_manager.clear()
        
        self.video_file.release()
        cv2.destroyAllWindows()

    def main_loop(self):
        target_delay = 1000 / (self.target_fps)

        fps_list = [self.target_fps] * 30

        while self.thread_manager.is_set():
            t_begin = time.time_ns()
            self.on_loop()
            t_end = time.time_ns()

            if self.frame_source_is_working is False:
                break

            t_delta = (t_end - t_begin) / 1e6
            delay = max(0, (target_delay - t_delta - 2) / 1000)
            if delay > 0.002:
                time.sleep(delay)

            
            # calculate the median fps.
            t_frame_end = time.time_ns()
            t_frame_delta = (t_frame_end - t_begin) / 1e6
            fps = round(1000 / t_frame_delta, 1)
            fps_list.append(fps)
            fps_list.pop(0)

            self.fps = np.median(fps_list)
            # print(self.fps)

            if self.display_frames:
                frame = Frame.resize(self.current_frame, target_h=300)

                if self.should_draw_fps:
                    frame = self.draw_fps(frame, int(self.fps))
                cv2.imshow(self.filename, frame)
                
                if (key := cv2.waitKey(1)) == ord('q'):
                    self.stop()

        self.stop()
