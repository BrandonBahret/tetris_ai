import numpy as np
import copy
import time

from Agent import PieceLUT_AOT as PieceLUT

from Types.NumbaDefinitions import TetrominoType, T1, np_tetromino

class Tetromino:
    def __init__(self, blocks_in:np.ndarray, orientation, piece_name, position=(-1, -1)):
        self.piece_name = piece_name
        self.blocks = blocks_in.copy()
        self.time_created = time.time()
        self.orientation = orientation
        self.set_position(position)
        self.is_held_piece = False

    def __eq__(self, other):
        if other is None:
            return False
        
        c1 = self.orientation == other.orientation
        c2 = self.position == other.position
        c3 = self.piece_name == other.piece_name
        return all([c1, c2, c3])

    @staticmethod
    def check_name_equality(p1, p2):
        if p1 and p2:
            return p1.piece_name == p2.piece_name
        elif p1 == p2:
            return True
        else:
            return False

    def copy(self):
        return copy.deepcopy(self)

    @property
    def position(self):
        return self._position

    def set_position(self, position):
        h, w = self.blocks.shape
        x, y = position
        self._position = position
        self.top_right = (x+w-1, y-h+1)

    def mark_as_held(self, is_held=True):
        self.is_held_piece = is_held
        
    @property
    def width(self):
        return self.blocks.shape[1]

    @property
    def height(self):
        return self.blocks.shape[0]

    @staticmethod
    def new_numba_tetromino_from_py(piece) -> T1:
        if piece is None:
            return None
        
        name = piece.piece_name
        orientation = piece.orientation
        position = piece.position
        is_held_piece = piece.is_held_piece

        np_piece = np.zeros(1, dtype=np_tetromino)
        np_piece['position'].fill(position)
        np_piece['name'].fill(name)
        np_piece['orientation'].fill(orientation)
        np_piece['is_held_piece'].fill(is_held_piece)
        return np_piece

    @staticmethod
    def numba_tetromino_to_py(np_piece:TetrominoType):
        name = np_piece['name'].item()
        orientation = np_piece['orientation'].item()
        position = np_piece['position'].item()
        is_held_piece = np_piece['is_held_piece'].item()
        piece_blocks = PieceLUT.get_piece_set(name)[orientation].copy()
        
        piece = Tetromino(piece_blocks, orientation, name, position)
        if is_held_piece:
            piece.mark_as_held()
        return piece
