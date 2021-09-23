import numpy as np
import time

from Types.Interface import InterfaceChecker


class FrameSourceInterface:
    "Interface Definition"
    def read(self): pass
    def get_title(self): pass

class FrameSourceBase:
    def __init__(self):
        self.child = super().__self__
        self.child_cls = super().__self_class__
        InterfaceChecker.check(self.child_cls, [FrameSourceInterface])

    def benchmark(self, test_length_seconds=60, trials=1):
        print("Preparing Benchmark...\n")

        fps_list = []
        for trial_idx in range(1, trials+1):
            frame_count = 0
            start_time = time.time()
            end_time = start_time + test_length_seconds

            print(f"Starting trial #{trial_idx}...")
            print(f"Capturing as many frames as possible in the next {test_length_seconds} seconds... Go!")
            while time.time() <= end_time:
                self.read()
                frame_count += 1

            fps_estimate = round(frame_count / test_length_seconds, 3)
            fps_list.append(fps_estimate)
            print(f"Trial {trial_idx} Results: {fps_estimate} FPS\n")
        
        print("")
        print("Done! Results:")
        print(f"\t-- median: {np.median(fps_list):0.2f} FPS")
        print(f"\t-- average: {np.mean(fps_list):0.2f} FPS")
