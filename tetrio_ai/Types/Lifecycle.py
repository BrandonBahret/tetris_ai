import numpy as np
import cv2
from threading import Event, Thread
import time

from Helpers.InputPool import InputPool
from Types.Interface import InterfaceChecker
from Types.Singleton import Singleton


class LifecycleInterface():
    "Interface specification"
    def on_loop(self, frame, fps): pass
    def on_destroy(self): pass
    def on_keypress(self, key): pass
    def get_frame(self): pass
    
class Lifecycle(metaclass=Singleton):

    def __init__(self):
        self.child = super().__self__
        self.child_cls = super().__self_class__
        InterfaceChecker.check(self.child_cls, [LifecycleInterface])

        self.thread_manager = None
        self.current_frame = None

    def imshow(self, title, frame, target_h=-1):
        if target_h != -1:
            ih, iw = (frame.shape[0], frame.shape[1])
            h = target_h
            w = h * iw // ih
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        cv2.imshow(title, frame)

    def on_destroy(self):
        if self.thread_manager is not None:
            self.thread_manager.clear()

    def run(self, target_fps=30):
        def main_loop():
            target_delay = 1000 / target_fps

            fps_list = [target_fps] * 30
            while self.thread_manager.is_set():

                t_begin = time.time_ns()
                success, frame = self.child.get_frame()
                if success is False:
                    break

                self.current_frame = frame.copy()
                fps = np.median(fps_list)

                # PerformanceChecker.check_performance(self.child.on_loop, self.current_frame, fps)
                self.child.on_loop(self.current_frame, fps)
                t_end = time.time_ns()
                

                ## Sleep to ensure the loop runs within the target_fps
                t_delta = (t_end - t_begin) / 1e6
                delay = max(0, (target_delay - t_delta - 2) / 1000)
                if delay > 0.001:
                    time.sleep(delay)

                # calculate the median fps.
                t_frame_end = time.time_ns()
                t_frame_delta = (t_frame_end - t_begin) / 1e6
                each_loop_fps = round(1000 / (t_frame_delta+.0001), 1)
                fps_list.append(each_loop_fps)
                fps_list.pop(0)

                if (last_key := cv2.waitKey(1)) != -1:
                    InputPool().add_key(last_key)

                for each_key in InputPool().get_key():
                    self.child.on_keypress(each_key)

                # while (each_key := InputPool().get_key()) is not None:
                #     self.child.on_keypress(each_key)

            self.child.on_destroy()

        self.thread_manager = Event()
        self.thread_manager.set()
        Thread(target=main_loop).start()
