import numpy as np
import numba as nb

from Agent.PieceClassifier import PieceClassifier
from Agent.Tetromino import Tetromino
# from Agent import PyMovePredict
from Agent import MovePrediction

from Controller.AgentController import AgentController
from Controller.MoveDescription import MoveDescription

from Helpers.PerformanceChecker import PerformanceChecker
from Observation.Gamestate import GamestateStruct
from Types.Singleton import Singleton

from Types.NumbaDefinitions import ModelWeights, GeneticWeightBundle
import pickle

def load_trained_model(number=0):
    name = "weights.pyobj" if number == 0 else f"weights #{number}.pyobj"
    with open(f"resources\\{name}", "rb+") as file:
        weight_sets = pickle.load(file)

    # weight_sets = [[1.]*5]*2

    bundled_weights = []
    for weights in weight_sets:
        bundled_weights.append(nb.typed.List([GeneticWeightBundle.new(-1.0, 1.0, w) for w in weights]))

    weights = ModelWeights.new(*bundled_weights)
    return weights

class GameAgent(metaclass=Singleton):
    '''Game Agent uses Gamestate to make and observe the consequences of its actions.'''

    def __init__(self):
        self._state:GamestateStruct = None
        self._target_move:MoveDescription = None
        self._controller = AgentController()
        self._weights = load_trained_model(0)

    # def wait_till_ready(self):
    #     mock_state = GamestateStruct()
    #     mock_state.active_piece = PieceClassifier().get_piece("T", 0)
    #     self.predict_move(mock_state)

    def process(self, state:GamestateStruct):
        self._state = state

        if state and state.new_active_piece:
            self._target_move = PerformanceChecker.check_performance(self.predict_move, state)
            PerformanceChecker().print_median_performances()

    def update(self):     
        if self._state is None:
            self._target_move = None
            self._controller.reset()

        if self._target_move and self._target_move.move_complete is False:
            active = self._state.active_piece
            self._controller.translate(active, self._target_move)

    @property
    def target_move(self):
        return self._target_move

    def predict_move(self, state:GamestateStruct):
        blocks = state.dead_blocks
        next_queue = nb.typed.List(state.next_queue)
        piece = state.active_piece
        np_piece = Tetromino.new_numba_tetromino_from_py(piece)

        held_piece = state.held_piece
        if held_piece:
            held_piece = held_piece.copy()
        np_held_piece = Tetromino.new_numba_tetromino_from_py(held_piece)
        is_held_disabled = state.is_held_disabled
        
        # result = PyMovePredict.predict_move(blocks, np_piece, np_held_piece, is_held_disabled, next_queue, self._weights)
        # target_move = MoveDescription.numba_move_to_py(result.move_list[0])
        target_move = MovePrediction.predict_move(blocks, np_piece, next_queue, np_held_piece, is_held_disabled, self._weights)
        target_move = MoveDescription.numba_move_to_py(target_move)
        return target_move
