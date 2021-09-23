import numpy as np
import cv2
from threading import Event, Thread
import time

from FrameGrabber.FrameSource.FrameSourceBase import FrameSourceBase

from Helpers.Draw import Draw
from Helpers.InputPool import InputPool


class FrameGrabber:

    def __init__(self, frame_source:FrameSourceBase, display_frames=True, should_draw_fps=False):
        self.display_frames = display_frames
        self.should_draw_fps = should_draw_fps
        self.frame_source = frame_source
        
        self.frame_source_is_working = True
        self.grabber_paused = False
        self.current_frame = None
        self.fps = -1
        self.target_fps = 60

        self.thread_manager = None

    def read(self):
        success = self.frame_source_is_working
        frame = self.current_frame
        if success:
            frame = frame.copy()
        return (success, frame)

    def on_loop(self):
        if self.grabber_paused:
            return None
    
        success, frame = self.frame_source.read()
        self.frame_source_is_working = success
        self.current_frame = frame

    def draw_fps(self, frame, fps, text_color=(255, 255, 255)):
        msg = "FPS: {}".format(fps)
        draw = Draw.begin(frame)
        Draw.text(draw, msg, (25, 25), text_color)
        return Draw.end(draw)

    def inc_next_frame(self):
        self.grabber_paused = True

        success, frame = self.frame_source.read()
        self.frame_source_is_working = success
        self.current_frame = frame

    def toggle_paused(self):
        self.grabber_paused = not self.grabber_paused

    def stop(self):
        if self.thread_manager is not None:
            self.thread_manager.clear()

    def main_loop(self):
        target_delay = 1000 / (self.target_fps)

        fps_list = [self.target_fps] * 20

        while self.thread_manager.is_set():

            t_begin = time.time_ns()
            # PerformanceChecker().check_performance(self.on_loop)
            self.on_loop()
            t_end = time.time_ns()

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
            # print(self.fps, np.mean(fps_list))

            if self.display_frames:
                if self.current_frame is None:
                    continue

                frame = self.current_frame
                # frame = Frame.resize(self.current_frame, target_h=400)
                if self.should_draw_fps:
                    frame = self.draw_fps(frame, int(self.fps))
                cv2.imshow(self.frame_source.get_title(), frame)
                if (last_key := cv2.waitKey(1)) != -1:
                    InputPool().add_key(last_key)

    def start(self, target_fps=60):
        self.target_fps = target_fps

        self.thread_manager = Event()
        self.thread_manager.set()
        self.on_loop()
        Thread(target=self.main_loop).start()
