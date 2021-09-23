import numpy as np

from Agent.Tetromino import Tetromino

from Types.Singleton import Singleton


class PieceClassifier(metaclass=Singleton):

    def __init__(self):
        self._pieces = {}

        T = np.array([
            [1, 1, 1],
            [0, 1, 0],
        ])
        self._pieces["T"] = [Tetromino(np.rot90(T, i), i, "T") for i in range(4)]

        J = np.array([
            [1, 1, 1],
            [0, 0, 1],
        ])
        self._pieces["J"] = [Tetromino(np.rot90(J, i), i, "J") for i in range(4)]

        L = np.array([
            [1, 1, 1],
            [1, 0, 0],
        ])
        self._pieces["L"] = [Tetromino(np.rot90(L, i), i, "L") for i in range(4)]

        S = np.array([
            [0, 1, 1],
            [1, 1, 0],
        ])
        self._pieces["S"] = [Tetromino(np.rot90(S, i), i, "S") for i in range(2)]

        Z = np.array([
            [1, 1, 0],
            [0, 1, 1],
        ])
        self._pieces["Z"] = [Tetromino(np.rot90(Z, i), i, "Z") for i in range(2)]

        O = np.array([
            [1, 1],
            [1, 1],
        ])
        self._pieces["O"] = [Tetromino(O, 0, "O")]

        I = np.array([
            [1, 1, 1, 1],
        ])
        self._pieces["I"] = [Tetromino(np.rot90(I, i), i, "I") for i in range(2)]

    def get_piece(self, piece_name, rotation_index=0):
        if piece_name is None:
            return None
        return self._pieces[piece_name][rotation_index].copy()

    def get_piece_set(self, piece_name):
        if piece_name is None:
            return None
        return [piece.copy() for piece in self._pieces[piece_name]]

    def match_piece(self, piece_blocks) -> Tetromino:
        for set_name, piece_set in self._pieces.items():
            for each_piece in piece_set:
                if np.array_equal(piece_blocks, each_piece.blocks):
                    return each_piece.copy()
        return None
