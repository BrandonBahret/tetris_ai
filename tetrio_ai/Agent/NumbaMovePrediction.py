import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from numba.pycc import CC
import numba as nb
import numpy as np

from Agent import NumbaTetromino
from Agent import NumbaMoveGenerator
from Agent import NumbaMoveFiltering
from Agent import NumbaBoardParsing as BoardParsing

from Helpers import ArrayFuncs

from typing import NamedTuple
from Types.NumbaDefinitions import NbExpose
from Types.NumbaDefinitions import T1
from Types.NumbaDefinitions import UnicodeListType
from Types.NumbaDefinitions import i1MatrixContType
from Types.NumbaDefinitions import IntArray
from Types.NumbaDefinitions import MoveDescriptionTuple
from Types.NumbaDefinitions import ConnectionLayer
from Types.NumbaDefinitions import ModelWeights
from Types.NumbaDefinitions import MoveScorePair, PredictionResult


cc = CC('MovePrediction')
# Uncomment the following line to print out the compilation steps
# cc.verbose = True

# __all__ = [
#     "branch_predict_move"
# ]

__all__ = [
    "predict_move"
]


@cc.export("find_block_peaks", IntArray(ConnectionLayer.lsttype))
@nb.njit(IntArray(ConnectionLayer.lsttype))
def find_block_peaks(col_layers):
    peaks = np.full((10), 22, dtype=np.int64)

    for col_layer in col_layers:
        x = col_layer.layer_index
        topmost_col_gap = list(col_layer.gaps_list[0])

        if len(topmost_col_gap) > 0:
            peaks[x] = 22 - (topmost_col_gap[-1]+1)
            
    return peaks

@cc.export("measure_flatness", nb.float64(ConnectionLayer.lsttype))
@nb.njit(nb.float64(ConnectionLayer.lsttype))
def measure_flatness(col_layers):
    '''measure flatness in terms of variance in block y positions'''
    peaks = find_block_peaks(col_layers)

    flatness = 1 - np.var(peaks) / 121
    ptp_score = 1 - np.ptp(peaks) / 22

    return (flatness + ptp_score) / 2

@cc.export("count_lines", nb.int8(i1MatrixContType))
@nb.njit(nb.int8(i1MatrixContType))
def count_lines(blocks):
    count = 0
    for row in blocks:
        count += np.sum(row) // 10
    
    return count

@cc.export("measure_layer_connectivity", nb.float64(ConnectionLayer.lsttype))
@nb.njit(nb.float64(ConnectionLayer.lsttype))
def measure_layer_connectivity(row_layers):
    connectivity_score = 0

    for layer in row_layers:
        connectivity = 10
        for gap in layer.gaps_list:
            connectivity -= len(gap)

        connectivity /= len(layer.gaps_list)
        connectivity_score += connectivity

    return connectivity_score / (9.0 * 23)

@cc.export("measure_aggregate_block_height", nb.int64(ConnectionLayer.lsttype))
@nb.njit(nb.int64(ConnectionLayer.lsttype))
def measure_aggregate_block_height(col_layers):
    peaks = BoardParsing.find_block_peaks(col_layers)
    return np.sum(peaks)

# @nb.njit(nb.int64(T1))
# def get_weight_index(piece):
#     piece_names = str("JLSZITO")
#     piece_name = str(piece.name.item())
#     return piece_names.index(piece_name)

@nb.njit(nb.int64(T1))
def get_weight_index(piece):
    piece_name = str(piece.name.item())
    piece_classes = [str("SZ"), str("JL"), str("I"), str("O"), str("T")]

    i = 0
    for i, p_class in enumerate(piece_classes):
        if piece_name in p_class:
            break

    return i

@cc.export("score_move", nb.float64(MoveDescriptionTuple.type, ModelWeights.type))
@nb.njit(nb.float64(MoveDescriptionTuple.type, ModelWeights.type))
def score_move(move, weights):
    piece = move.destinations[0]
    weight_index = get_weight_index(piece)

    col_layers = BoardParsing.parse_row_layers(move.target_outcome.T)
    row_layers = BoardParsing.parse_row_layers(move.target_outcome)

    line_total = count_lines(move.target_placement_simple)
    flatness = measure_flatness(col_layers)
    row_connectivity = measure_layer_connectivity(row_layers)
    aggregate_height = measure_aggregate_block_height(col_layers)

    if line_total > 0:
        score = line_total
        score += flatness
        score += (row_connectivity * weights.c0row_connectivity[weight_index].value)
        score -= aggregate_height

    else:
        score = line_total
        score += flatness
        score += (row_connectivity * weights.c1row_connectivity[weight_index].value)
        score -= aggregate_height

    return score

@cc.export("find_best_move", MoveScorePair.type(MoveDescriptionTuple.lsttype, ModelWeights.type))
@nb.njit(MoveScorePair.type(MoveDescriptionTuple.lsttype, ModelWeights.type))
def find_best_move(moves, weights):
    best_move = moves[0]
    best_score = score_move(best_move, weights)

    for move in moves[1:]:
        each_score = score_move(move, weights)

        if each_score > best_score:
            best_score = each_score
            best_move = move

    return MoveScorePair.new(best_move, best_score)

@cc.export("predict_move", MoveDescriptionTuple.type(i1MatrixContType, T1, nb.types.ListType(nb.types.unicode_type), nb.types.Optional(T1), nb.boolean, ModelWeights.type))
@nb.njit(MoveDescriptionTuple.type(i1MatrixContType, T1, nb.types.ListType(nb.types.unicode_type), nb.types.Optional(T1), nb.boolean, ModelWeights.type))
def predict_move(blocks, np_piece, next_queue, np_held_piece, is_held_disabled, weights):
    # intial_bundle = NumbaMoveFiltering.parse_metrics(blocks)

    moves = NumbaMoveGenerator.generate_moves(blocks, np_piece.copy())
    # moves = NumbaMoveFiltering.filter_moves(moves, intial_bundle, 15, weights)
    target_move, target_score = find_best_move(moves, weights)

    if is_held_disabled:
        return target_move

    else:
        if np_held_piece is None:
            np_held_piece = NumbaTetromino.new_tetromino(next_queue[0], 0, (4, 0), True)

        held_moves = NumbaMoveGenerator.generate_moves(blocks, np_held_piece.copy())
        # held_moves = NumbaMoveFiltering.filter_moves(held_moves, intial_bundle, 15, weights)
        held_target_move, held_target_score = find_best_move(held_moves, weights)

        if held_target_score > target_score:
            return held_target_move

        return target_move
        
# @cc.export("get_mock_result", PredictionResult.type())
# @nb.njit(PredictionResult.type())
# def get_mock_result():
#     mock_destinations = nb.typed.List.empty_list(T1)
#     mock_blocks = np.zeros((2,2), dtype=np.int8)
#     mock_move = MoveDescriptionTuple.new(mock_destinations, mock_blocks, mock_blocks, mock_blocks)
#     mock_score = -np.inf
#     mock_scored_pair = MoveScorePair.new(mock_move, mock_score)
#     mock_move_list = nb.typed.List.empty_list(MoveDescriptionTuple.type)
#     return PredictionResult.new(mock_scored_pair, mock_move_list)

# rettype = PredictionResult.type
# predict_move_sig = rettype(i1MatrixContType, T1, UnicodeListType, ModelWeights.type, nb.int64, nb.int64, MoveScorePair.type, MoveDescriptionTuple.lsttype)
# @cc.export("branch_predict_move", predict_move_sig)
# @nb.njit(predict_move_sig, nogil=True)
# def branch_predict_move(blocks, active_piece, next_queue, weights, step, look_ahead, best_move, move_list):
#     last_step = min(look_ahead, len(next_queue)-1)

#     next_piece = NumbaTetromino.new_tetromino(next_queue[step], 0, (4, 2), False)
#     best_score = lambda pair: -np.inf if pair is None else pair.score

#     bundle = NumbaMoveFiltering.parse_metrics(blocks)
#     moves = NumbaMoveGenerator.generate_moves(blocks, active_piece.copy())
#     moves = NumbaMoveFiltering.filter_moves(moves, bundle, 8, weights)

#     if step < last_step:
#         for move in moves:
#             initial_score = best_score(best_move)
#             result = branch_predict_move(move.target_outcome, next_piece, next_queue, weights, step+1, look_ahead, best_move, move_list)

#             if result is not None:
#                 if result.best_move.score > initial_score:
#                     best_move = result.best_move

#                     if step == last_step -1:
#                         move_list.clear()
#                         move_list.append(best_move.move)
                    
#                     move_list.insert(0, move)
            
#     else:
#         iters_best = find_best_move(moves, weights)
#         if iters_best.score > best_move.score:
#             best_move = iters_best

#     return PredictionResult.new(best_move, move_list)


if __name__ == "__main__":
    cc.compile()
