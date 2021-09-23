import numba as nb
import numpy as np

import pickle

from Agent import NumbaTetromino
from Agent import PieceLUT

from Agent import NumbaMovePrediction
from Agent import NumbaMoveFiltering
from Agent import NumbaBoardParsing

from typing import NamedTuple
from Types.NumbaDefinitions import NbExpose
from Types.NumbaDefinitions import i1MatrixContType
from Types.NumbaDefinitions import i1MatrixAnyType
from Types.NumbaDefinitions import i1MatrixTypes
from Types.NumbaDefinitions import MoveDescriptionTuple
from Types.NumbaDefinitions import T1
from Types.NumbaDefinitions import ModelWeights, GeneticWeightBundle


np_unichar = np.dtype('<U1')
UnicharType = nb.from_dtype(np_unichar)
UnicharListType = nb.types.ListType(UnicharType)

@nb.njit(UnicharType[:]())
def piece_names():
    arr = np.zeros(shape=(7,), dtype=np_unichar)
    for i, name in enumerate("TJLSZIO"):
        arr[i] = name

    return arr

@nb.experimental.jitclass
class SevenBag:
    __inputs: nb.types.ListType(nb.types.unicode_type)
    __input_i: nb.int64
    __input_len: nb.int64
    next_queue: nb.types.ListType(nb.types.unicode_type)

    def __init__(self, inputs):
        self.__inputs = inputs
        self.__input_len = len(inputs)
        self.__input_i = 0
        
        self.next_queue = nb.typed.List([self.__read_input() for c in range(5)])

    def __read_input(self):
        next_piece = self.__inputs[self.__input_i]
        self.__input_i = (self.__input_i + 1) % self.__input_len
        return str(next_piece)

    def get_next_piece(self):
        next_piece_name = self.next_queue.pop(0)
        new_piece_name = self.__read_input()
        self.next_queue.append(new_piece_name)

        next_piece = NumbaTetromino.new_tetromino(next_piece_name, 0, (4, 2), False)
        return next_piece

    def get_piece(self, name):
        next_piece = NumbaTetromino.new_tetromino(name, 0, (4, 2), False)
        return next_piece


@nb.njit(i1MatrixAnyType(T1))
def get_blocks(piece):
    name = piece.name.item()
    orientation = piece.orientation.item()
    varient_set = PieceLUT.get_piece_set(name)

    return varient_set[orientation]

@nb.njit(nb.boolean(MoveDescriptionTuple.type))
def should_hold_piece(move):
    d1 = move.destinations[-1]
    return NumbaTetromino.get_tetromino_is_held(d1)

@nb.njit(nb.boolean(i1MatrixContType))
def is_gameover(blocks):
    return np.any(blocks[:1] > 0)


@NbExpose
class SimulationBundle(NamedTuple):
    gameover: nb.boolean
    hold_count: nb.int64
    line_count: nb.int64
    final_board_height: nb.int64
    created_hole_count: nb.int64
    tetris_count: nb.int64
    iter_count: nb.int64
    max_iters: nb.int64

@nb.njit(SimulationBundle.type(i1MatrixContType, nb.int64, ModelWeights.type, nb.types.ListType(nb.types.unicode_type)))
def simulate(blocks, iterations, weights, inputs):
    gameover = False
    line_count = 0
    tetris_count = 0
    hold_count = 0

    bag = SevenBag(inputs)
    held_piece = None
    is_held_disabled = False
    created_hole_count = 0

    for i in range(iterations):
        if is_gameover(blocks):
            gameover = True
            break

        active_piece = bag.get_next_piece()
        next_queue = bag.next_queue

        target_move = NumbaMovePrediction.predict_move(blocks, active_piece, next_queue, held_piece, is_held_disabled, weights)

        if should_hold_piece(target_move):
            is_held_disabled = True
            hold_count += 1
            
            if held_piece is None:
                held_piece = active_piece.copy()
                NumbaTetromino.set_tetromino_is_held(held_piece)
                continue
            else:
                active_piece, held_piece = held_piece.copy(), active_piece.copy()
                NumbaTetromino.set_tetromino_is_held(held_piece)
                active_piece.is_held_piece.fill(False)
                target_move = NumbaMovePrediction.predict_move(blocks, active_piece, next_queue, held_piece, is_held_disabled, weights)

        else:
            is_held_disabled = False

        initial_bundle = NumbaMoveFiltering.parse_metrics(blocks)
        bundle = NumbaMoveFiltering.parse_metrics(target_move.target_outcome)
        created_hole_count += max(0, (bundle.hole_count - initial_bundle.hole_count))
        blocks = target_move.target_outcome

        lines = NumbaMovePrediction.count_lines(target_move.target_placement_simple)
        line_count += lines
        tetris_count += lines//4

    col_layers = NumbaBoardParsing.parse_row_layers(blocks.T)
    final_board_height = NumbaMovePrediction.measure_aggregate_block_height(col_layers)

    return SimulationBundle.new(gameover, hold_count, line_count, final_board_height, created_hole_count, tetris_count, i+1, iterations)
