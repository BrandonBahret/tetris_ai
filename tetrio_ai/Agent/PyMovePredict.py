import numpy as np
import numba as nb

from Agent import NumbaTetromino
from Types.NumbaDefinitions import MoveScorePair, PredictionResult
from Agent import MovePrediction
# from Agent import NumbaMoveGenerator as MoveGenerator
# from Agent import NumbaMoveFiltering as MoveFiltering
from Agent import MoveGenerator
from Agent import MoveFiltering

from Types.NumbaDefinitions import T1
from Types.NumbaDefinitions import MoveDescriptionTuple

from concurrent.futures import ThreadPoolExecutor as executor


def get_mock_result():
    mock_destinations = nb.typed.List.empty_list(T1)
    mock_blocks = np.zeros((2,2), dtype=np.int8)
    mock_move = MoveDescriptionTuple.new(mock_destinations, mock_blocks, mock_blocks, mock_blocks)
    mock_score = -np.inf
    mock_scored_pair = MoveScorePair.new(mock_move, mock_score)
    mock_move_list = nb.typed.List.empty_list(MoveDescriptionTuple.type)
    return PredictionResult.new(mock_scored_pair, mock_move_list)

def impl_predict_move(blocks, active_piece, next_queue, weights, look_ahead):
    next_piece = NumbaTetromino.new_tetromino(next_queue[0], 0, (4, 2), False)

    bundle = MoveFiltering.parse_metrics(blocks)
    moves = MoveGenerator.generate_moves(blocks, active_piece.copy())
    moves = MoveFiltering.filter_moves(moves, bundle, 8, weights)

    sentinal_move = get_mock_result().best_move

    results = []
    with executor(max_workers=8) as ex:
        for i in range(8):
            move_list = nb.typed.List.empty_list(MoveDescriptionTuple.type)
            move_blocks = moves[i].target_outcome
            future = ex.submit(MovePrediction.branch_predict_move, move_blocks, next_piece, next_queue, weights, 1, look_ahead, sentinal_move, move_list)
            results.append((moves[i], future))

    mapped_results = []
    for move, future in results:
        result = future.result()
        result.move_list.insert(0, move)
        mapped_results.append(result)

    return max(mapped_results, key= lambda r: r.best_move.score)

def predict_move(blocks, active_piece, held_piece, is_held_disabled, next_queue, weights, look_ahead=2):
    active_result = impl_predict_move(blocks.copy(), active_piece, next_queue, weights, look_ahead)

    if is_held_disabled:
        return active_result

    else:
        is_swap_move = True
        if held_piece is None:
            is_swap_move = False
            held_piece = NumbaTetromino.new_tetromino(next_queue[0], 0, (4, 0), True)

        held_result = impl_predict_move(blocks.copy(), held_piece, next_queue, weights, look_ahead)

        if held_result.best_move.score > active_result.best_move.score:
            if is_swap_move:
                d1 = held_result.move_list[0].destinations[0]
                held_result.move_list[0].destinations.insert(0, d1)
            
            return held_result

        return active_result
