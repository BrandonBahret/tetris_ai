import numpy as np
import copy
import time

from Agent.Tetromino import Tetromino

from FrameGrabber.FrameStacker import FrameStacker

from Observation.ActivePieceTracker import ActivePieceTracker
from Observation.BlockDetector import BlockDetector
from Observation.ImageProc import ImageProc
from Observation.ContainerDetector import ContainerDetector
from Observation.Gamestate import GamestateStruct
from Observation.PieceQueueParser import PieceQueueParser
from Observation.HeldPieceParser import HeldPieceParser

import Helpers.Geometry as Geo
from Types.Singleton import Singleton
from Types.Interface import InterfaceChecker


class PreprocessedFrame:
    def __init__(self, frame, masks=None):
        self.frame = frame.copy()
        self.masks = masks
        if self.masks is None:
            self.masks = ImageProc.segment(frame)

    def copy(self):
        return copy.deepcopy(self)

class GamestateSynthInterface:
    "Interface Definition"
    def process(self, frame): pass
    def construct_state(self): pass

class GamestateSynth(metaclass=Singleton):
    '''Gamestate Synthesizer is used to construct a representation of the game'''
    
    def __init__(self):
        InterfaceChecker.check(self.__class__, [GamestateSynthInterface])

        ## Initialize detector objects
        self.container_detector = ContainerDetector()
        self.block_segmenter = ActivePieceTracker()
        self.block_detector = None
        self.next_queue_parser = None
        self.held_piece_parser = None

        ## Initialize state objects
        self.state = GamestateStruct()
        self.previous_state = None
        self.new_active_piece = False
        self.active_piece_start_t = time.time()
        self.game_start_t = time.time()

        ## Register frame stackers with each detector callback
        self.container_detector_stacker = FrameStacker(self.container_detector.process, ignore_period=1*50, stack_length=5)

    def process(self, frame):
        preprocessed_frame = PreprocessedFrame(frame)
        if not self.container_detector.has_feature():
            self.container_detector_stacker.stack_frame(preprocessed_frame)

        self.block_detector_callback(preprocessed_frame)
        # PerformanceChecker.check_performance(self.block_detector_callback, preprocessed_frame)

    def block_detector_callback(self, preprocessed_frame):
        container_lines, rois = self.container_detector.construct_feature()

        if len(container_lines) != 3:
            return None

        gameboard_roi = (rois["game"][0], rois["sky"][1])
        gameboard_width = Geo.Rectangle.width(gameboard_roi)
        block_length = (gameboard_width / 10) * 1.00675

        self.next_queue_parser = PieceQueueParser(rois["next"], block_length)
        self.next_queue_parser.process(preprocessed_frame)

        self.held_piece_parser = HeldPieceParser(rois["held"], block_length)
        self.held_piece_parser.process(preprocessed_frame)

        self.block_detector = BlockDetector(gameboard_roi, block_length)
        self.block_detector.process(preprocessed_frame)

        blocks = self.block_detector.construct_feature()
        self.block_segmenter.process(blocks)

    def get_active_piece_time(self):
        ## time how long the current active piece has been in play
        if self.previous_state is not None:
            self.new_active_piece = False

            current_mass = np.sum(self.state.dead_blocks)
            previous_mass = np.sum(self.previous_state.dead_blocks)
            if previous_mass != current_mass:
                self.active_piece_start_t = time.time()
                self.new_active_piece = True

            elif self.state.next_queue != self.previous_state.next_queue:
                self.active_piece_start_t = time.time()
                self.new_active_piece = True

            elif not Tetromino.check_name_equality(self.state.held_piece, self.previous_state.held_piece):
                self.active_piece_start_t = time.time()
                self.new_active_piece = True
        else:
            self.new_active_piece = True

        return time.time() - self.active_piece_start_t

    def construct_state(self):
        game_lifetime = time.time() - self.game_start_t
        if game_lifetime < 4:
            self.state.dead_blocks.fill(0)
            return None

        blocks_components = self.block_segmenter.construct_feature()
        if blocks_components is None:
            self.game_start_t = time.time()
            return None

        self.state.dead_blocks = blocks_components[0]
        self.state.active_piece = blocks_components[1]
        self.state.next_queue = self.next_queue_parser.construct_feature()
        held_piece = self.held_piece_parser.construct_feature()
        self.state.held_piece = held_piece.piece 
        self.state.is_held_disabled = held_piece.is_held_disabled 
        self.state.active_piece_lifetime = self.get_active_piece_time()
        self.state.new_active_piece = self.new_active_piece
        self.state.game_lifetime = game_lifetime
        self.previous_state = self.state.copy()

        if self.state.next_queue is None or len(self.state.next_queue) != 5:
            self.state = GamestateStruct()
            self.previous_state = None
            return None

        return self.state
