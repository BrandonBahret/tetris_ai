import numpy as np
import copy

from Agent.Tetromino import Tetromino


class GamestateStruct:
    def __init__(self):
        self.active_piece:Tetromino = None
        self.dead_blocks:np.array = np.zeros((23, 10), dtype=np.int8)
        self.next_queue:list = list("JSTLZ")
        self.held_piece:Tetromino = None
        self.is_held_disabled:bool = False
        self.active_piece_lifetime:int = 0
        self.new_active_piece:bool = False
        self.game_lifetime:int = 0

    def copy(self):
        return copy.deepcopy(self)
