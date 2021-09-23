import numpy as np
import numba as nb

from Types.NumbaDefinitions import IntPair
from Types.NumbaDefinitions import UnicharType
from Types.NumbaDefinitions import T1, np_tetromino

__all__ = [
    "new_tetromino",
    "get_tetromino_position",
    "set_tetromino_position",
    "set_tetromino_name",
    "set_tetromino_orientation"
]


@nb.njit(T1(UnicharType, nb.int8, IntPair, nb.boolean))
def new_tetromino(name, orientation, position, is_held_piece):
    np_piece = np.zeros(1, dtype=np_tetromino)
    
    x, y = position
    np_piece.position.x.fill(x)
    np_piece.position.y.fill(y)

    np_piece.name.fill(name)
    np_piece.orientation.fill(orientation)
    np_piece.is_held_piece.fill(is_held_piece)

    return np_piece

@nb.njit(nb.boolean(T1, T1))
def check_equality(p1, p2):
    if p1.position.x.item() != p2.position.x.item():
        return False
    if p1.position.y.item() != p2.position.y.item():
        return False
    if p1.name.item() != p2.name.item():
        return False
    if p1.orientation.item() != p2.orientation.item():
        return False
    if p1.is_held_piece.item() != p2.is_held_piece.item():
        return False

    return True

@nb.njit(IntPair(T1))
def get_tetromino_position(np_piece):
    x = np_piece.position.x.item()
    y = np_piece.position.y.item()
    return (x, y)

@nb.njit(nb.void(T1, IntPair))
def set_tetromino_position(np_piece, position):
    x, y = position
    np_piece.position.x.fill(x)
    np_piece.position.y.fill(y)

@nb.njit(nb.void(T1, UnicharType))
def set_tetromino_name(np_piece, name):
    np_piece.name.fill(name)

@nb.njit(nb.void(T1, nb.int8))
def set_tetromino_orientation(np_piece, orientation):
    np_piece.orientation.fill(orientation)

@nb.njit(nb.int8(T1))
def get_tetromino_orientation(np_piece):
    return np_piece.orientation.item()

@nb.njit(nb.void(T1))
def set_tetromino_is_held(np_piece):
    np_piece.is_held_piece.fill(True)

@nb.njit(nb.boolean(T1))
def get_tetromino_is_held(np_piece):
    return np_piece.is_held_piece.item()
