import numpy as np

class FrameSequence:

    def __init__(self, sequence_length):
        self.sequence = [None] * sequence_length

    def hstack_sequence(self):
        height = self.sequence[-1].shape[0]
        padding = np.full((height, 1), 128, np.uint8)

        frames = tuple([np.hstack((frame, padding)) for frame in self.get_sequence()])
        return np.hstack(frames)
    
    def add_frame(self, frame):
        self.sequence.pop(0)
        self.sequence.append(frame)

    def get_sequence(self):
        return [f for f in self.sequence if f is not None]
