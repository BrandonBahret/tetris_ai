import cv2
import os

from Agent.GameAgent import GameAgent
from Observation.GamestateSynth import GamestateSynth

from FrameGrabber.FrameGrabber import FrameGrabber
from FrameGrabber.FrameSource import D3DSource, VideoSource

from Helpers.PerformanceChecker import PerformanceChecker
from GamestateRenderer import GamestateRenderer
from Types.Lifecycle import Lifecycle


class app(Lifecycle):
    '''object responsible for processing the application lifecycle.'''

    def __init__(self):
        super().__init__()
        self.app_title = "TETR.IO"

        self.gamestate_synth = GamestateSynth()
        self.gamestate_renderer = GamestateRenderer()
        self.gameagent = GameAgent()
        # self.gameagent.wait_till_ready()

        # self.frame_source = VideoSource("proj_files/snaps/tet new.avi", starting_frame=0)
        self.frame_source = D3DSource("TETR.IO")

        self.frame_grabber = FrameGrabber(self.frame_source, display_frames=False)
        self.frame_grabber.start(target_fps=30)

    def get_frame(self):
        success, frame = self.frame_grabber.read()
        return (success, frame)

    def on_loop(self, frame, fps):
        self.gamestate_synth.process(frame)
        gamestate = self.gamestate_synth.construct_state()

        self.gameagent.process(gamestate)
        if self.frame_grabber.frame_source.__class__ is D3DSource:
            self.gameagent.update()

        target_move = self.gameagent.target_move
        self.gamestate_renderer.process(gamestate, target_move, fps)

    def on_keypress(self, key):
        if key == ord('r'):
            self.frame_source.reset()

        elif key == ord('o'):
            path = os.getcwd()
            PerformanceChecker().save_performance_data(f"{path}/data.csv")

        elif key == ord('s'):
            self.frame_grabber.toggle_paused()

        elif key == ord('n'):
            self.frame_grabber.inc_next_frame()

        elif key == ord('q'):
            self.on_destroy()
        
        elif key == ord('p'):
            cv2.imwrite("proj_files/snaps/tetris_new.png", self.current_frame)
        
    def on_destroy(self):
        super().on_destroy()
        
        # Release everything if job is finished
        self.frame_grabber.stop()
        self.gamestate_renderer.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app().run(target_fps = 30)
