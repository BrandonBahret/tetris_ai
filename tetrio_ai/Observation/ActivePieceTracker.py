import numpy as np

from Agent.PieceClassifier import PieceClassifier

from FrameGrabber.FrameSequence import FrameSequence
from FrameGrabber.Frame import Frame

from Types.Singleton import Singleton


class ActivePieceTracker(metaclass=Singleton):
    
    def __init__(self):
        self.blocks_sequence = FrameSequence(2)
        self.active_blocks = None
        self.dead_blocks = None

    def process(self, blocks):
        if blocks is None or blocks.shape != (23, 10):
            return None

        self.blocks_sequence.add_frame(blocks)
        sequence = self.blocks_sequence.get_sequence()
        if len(sequence) != 2:
            return None
        
        current_blocks = sequence[-1]
        previous_blocks = sequence[-2]

        ## Process blocks array
        block_difference = int(np.sum(current_blocks)) - int(np.sum(previous_blocks))
        if block_difference >= 4 or block_difference < 0:
            cur_skyblocks = self.grab_skyblocks(current_blocks)
            pre_skyblocks = self.grab_skyblocks(previous_blocks)

            if Frame.compare_shape(cur_skyblocks, pre_skyblocks) is False:
                return None

            changed_skyblocks = cur_skyblocks & ~pre_skyblocks
            if np.sum(changed_skyblocks) != 4:
                changed_skyblocks = cur_skyblocks & ~(pre_skyblocks & ~cur_skyblocks)

            if np.sum(changed_skyblocks) == 4:
                dead_blocks = blocks & ~changed_skyblocks
                self.dead_blocks = dead_blocks

        if Frame.compare_shape(blocks, self.dead_blocks):
            self.active_blocks = blocks & ~self.dead_blocks


    def grab_skyblocks(self, blocks):
        skyblocks = blocks.copy()
        skyblocks[3:, :].fill(0)
        return skyblocks

    def find_active_piece(self):
        if self.active_blocks is None:
            return None

        if np.max(self.active_blocks) == 0:
            return None

        ys, xs = np.where(self.active_blocks >= 1)
        ys, xs = (ys[:4], xs[:4])
        origin = (np.min(xs), np.max(ys))
        top_right = (np.max(xs), np.min(ys))
        active_piece_roi = (origin, top_right)

        active_blocks = Frame.crop(self.active_blocks, active_piece_roi)
        active_piece = PieceClassifier().match_piece(active_blocks)
        if active_piece is not None:
            active_piece.set_position(origin)

        return active_piece

    def construct_feature(self):
        active_piece = self.find_active_piece()
        if active_piece is None or self.dead_blocks is None:
            return None

        return (self.dead_blocks, active_piece)
