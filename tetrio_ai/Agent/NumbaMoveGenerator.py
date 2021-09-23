import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import numba as nb
from numba.pycc import CC

from Agent import NumbaBoardParsing as BoardParsing
from Agent import NumbaMoveFiltering as MoveFiltering
from Agent import NumbaTetromino
from Agent import PieceLUT

from Types.NumbaDefinitions import i1MatrixContType
from Types.NumbaDefinitions import i1MatrixAnyType

from Types.NumbaDefinitions import MoveDescriptionTuple
from Types.NumbaDefinitions import ConnectionLayer
from Types.NumbaDefinitions import OverhangSet

from Types.NumbaDefinitions import IntSet, IntSetList
from Types.NumbaDefinitions import T1


cc = CC('MoveGenerator')
# Uncomment the following line to print out the compilation steps
# cc.verbose = True

place_sig = nb.types.Optional(i1MatrixContType)(i1MatrixAnyType, T1, i1MatrixAnyType)
@cc.export("place_piece", place_sig)
@nb.njit(place_sig)
def place_piece(blocks, np_piece, piece_blocks):
    blocks = blocks.copy()
    h, w = piece_blocks.shape
    x, y = NumbaTetromino.get_tetromino_position(np_piece)

    for row_i, layer in enumerate(piece_blocks):
        # print(row_i, layer)
        for col_i, col_value in enumerate(layer):
            py, px = (row_i+(y-h+1), col_i+x)
            if px >= 10 or py >= 23:
                return None

            # print((row_i+y, col_i+x), (px, py), col_value)
            if py >= 0 and px >= 0:
                blocks[py, px] += col_value

    return blocks


clear_sig = i1MatrixContType(i1MatrixContType)
@cc.export("clear_lines", clear_sig)
@nb.njit(clear_sig)
def clear_lines(blocks):
    blocks = blocks.copy()
    for i, row in enumerate(blocks):
        if np.sum(row) == 10:
            blocks[1:i+1] = blocks[0:i]
            blocks[0] = 0
    
    return blocks

drop_sig = nb.types.Optional(MoveDescriptionTuple.type)(i1MatrixAnyType, T1, i1MatrixAnyType)
@cc.export("drop_piece", drop_sig)
@nb.njit(drop_sig)
def drop_piece(blocks, np_piece, piece_blocks):
    np_piece = np_piece.copy()
    x, y = NumbaTetromino.get_tetromino_position(np_piece)

    target_placement = place_piece(blocks, np_piece, piece_blocks)
    if target_placement is None:
        return None
    if not np.all(np.asarray(target_placement) <= 1):
        return None

    while target_placement is not None:      
        # while piece position is valid, lower piece
        if not np.all(np.asarray(target_placement) <= 1):
            break

        x, y = NumbaTetromino.get_tetromino_position(np_piece)
        NumbaTetromino.set_tetromino_position(np_piece, (x, y+1))
        target_placement = place_piece(blocks, np_piece, piece_blocks)

    # raise piece up one unit to make position valid
    x, y = NumbaTetromino.get_tetromino_position(np_piece)
    NumbaTetromino.set_tetromino_position(np_piece, (x, y-1))

    target_placement_simple = place_piece(blocks, np_piece, piece_blocks)
    target_outcome = clear_lines(target_placement_simple)
    target_placement = place_piece(target_placement_simple, np_piece, piece_blocks)
    # target_placement[y-1, x] = 3

    destinations = nb.typed.List([np_piece])
    return MoveDescriptionTuple.new(destinations, target_placement.copy(), target_outcome.copy(), target_placement_simple.copy())

### tuck move generation
is_empty_sig = nb.boolean(IntSet, IntSetList)
@cc.export("is_space_empty", is_empty_sig)
@nb.njit(is_empty_sig)
def is_space_empty(space_between, gaps_list):
    for gap in gaps_list:
        if space_between.issubset(gap):
            return True

    return False


horizontal_path_sig = nb.boolean(ConnectionLayer.lsttype, T1, T1, i1MatrixAnyType)
@cc.export("is_horizontal_path_clear", horizontal_path_sig)
@nb.njit(horizontal_path_sig)
def is_horizontal_path_clear(row_layers, from_piece, to_piece, piece_blocks):
    fx, fy = NumbaTetromino.get_tetromino_position(from_piece)
    tx, ty = NumbaTetromino.get_tetromino_position(to_piece)
    h, w = piece_blocks.shape

    if fx > tx:
        tmp_x, tmp_y = fx, fy
        fx, fy = tx, ty
        tx, ty = tmp_x, tmp_y

    for layer_i, y in enumerate(range(fy - h+1, fy+1)):
        piece_layer = piece_blocks[layer_i]
        occupied_xs = np.where(piece_layer != 0)[0]
        leftmost_x = occupied_xs[0]

        # Find the space between (from & to) at each layer.
        space_between = set(range(leftmost_x + fx  ,  leftmost_x + tx+1))

        if not is_space_empty(space_between, row_layers[y].gaps_list):
            return False
    
    return True

vertical_path_sig = nb.boolean(ConnectionLayer.lsttype, T1, T1, i1MatrixAnyType)
@cc.export("is_vertical_path_clear", vertical_path_sig)
@nb.njit(vertical_path_sig)
def is_vertical_path_clear(col_layers, from_piece, to_piece, piece_blocks):
    fx, fy = NumbaTetromino.get_tetromino_position(from_piece)
    tx, ty = NumbaTetromino.get_tetromino_position(to_piece)
    h, w = piece_blocks.shape

    if fy > ty:
        tmp_x, tmp_y = fx, fy
        fx, fy = tx, ty
        tx, ty = tmp_x, tmp_y

    for layer_i, x in enumerate(range(fx, fx+w)):
        piece_layer = piece_blocks.T[layer_i]
        occupied_ys = np.where(piece_layer != 0)[0]
        topmost_y = occupied_ys[0]

        # Find the space between (from & to) at each layer.
        space_between = set(range(fy, topmost_y + ty-1))

        if not is_space_empty(space_between, col_layers[x].gaps_list):
            return False
    
    return True


path_clear_sig = nb.boolean(ConnectionLayer.lsttype, ConnectionLayer.lsttype, T1, T1, i1MatrixAnyType)
@cc.export("is_path_clear", path_clear_sig)
@nb.njit(path_clear_sig)
def is_path_clear(col_layers, row_layers, from_piece, to_piece, piece_blocks):
    fx, fy = NumbaTetromino.get_tetromino_position(from_piece)
    tx, ty = NumbaTetromino.get_tetromino_position(to_piece)

    if fy == ty:
        return is_horizontal_path_clear(row_layers, from_piece, to_piece, piece_blocks)

    if fx == tx:
        return is_vertical_path_clear(col_layers, from_piece, to_piece, piece_blocks)

    return False

placement_possible_sig = nb.boolean(i1MatrixContType, T1, i1MatrixAnyType)
@cc.export("is_placement_possible", placement_possible_sig)
@nb.njit(placement_possible_sig)
def is_placement_possible(blocks, np_piece, piece_blocks):
    placement = place_piece(blocks, np_piece, piece_blocks)
    if placement is None:
        return False
    
    if np.any(np.asarray(placement) > 1):
        return False

    return True

tuck_sig = nb.types.Optional(MoveDescriptionTuple.type)(i1MatrixContType, ConnectionLayer.lsttype, ConnectionLayer.lsttype, OverhangSet.type, T1, T1)
@cc.export("tuck_piece", tuck_sig)
@nb.njit(tuck_sig)
def tuck_piece(blocks, col_layers, row_layers, overhang, destination_piece, active_piece):
    varient_set = PieceLUT.get_piece_set(active_piece.name.item())

    destination_orientation = destination_piece.orientation.item()
    destination_blocks = varient_set[destination_orientation]
    dx, dy = NumbaTetromino.get_tetromino_position(destination_piece)
    dh, dw = destination_blocks.shape

    mouth_width = len(overhang.mouth.gap_set)
    if not dw <= mouth_width:
        return None

    if not is_placement_possible(blocks, destination_piece, destination_blocks):
        return None

    destinations = nb.typed.List.empty_list(T1)

    # Find the first destination
    d1_piece = destination_piece.copy()
    mouth_x = min(overhang.mouth.gap_set)
    NumbaTetromino.set_tetromino_position(d1_piece, (mouth_x, dy))

    if is_placement_possible(blocks, d1_piece, destination_blocks):
        destinations.append(d1_piece.copy())
    else:
        return None
    

    # Change the destination piece if it needs to drop into place
    destation_drop = drop_piece(blocks, destination_piece, destination_blocks)
    if destation_drop is not None:
        final_placement = destation_drop.destinations[0]

        are_equal = NumbaTetromino.check_equality(final_placement, destination_piece)
        if not are_equal:
            destination_piece = final_placement.copy()

    # Ensure there is a clear path from D1 and the Destination
    path_clear = is_path_clear(col_layers, row_layers, d1_piece, destination_piece, destination_blocks)
    if not path_clear:
        return None
        
    destinations.append(destination_piece.copy())

    target_placement_simple = place_piece(blocks, destination_piece, destination_blocks)
    target_placement = place_piece(target_placement_simple, destination_piece, destination_blocks)
    target_outcome = clear_lines(target_placement_simple)

    return MoveDescriptionTuple.new(destinations, target_placement.copy(), target_outcome.copy(), target_placement_simple.copy())

### -----
append_unique_sig = nb.void(MoveDescriptionTuple.lsttype, MoveDescriptionTuple.type)
@cc.export("append_unique_move", append_unique_sig)
@nb.njit(append_unique_sig)
def append_unique_move(moves, new_move):
    new_position = NumbaTetromino.get_tetromino_position(new_move.destinations[-1])

    for each_move in moves:
        each_position = NumbaTetromino.get_tetromino_position(each_move.destinations[-1])

        if each_position == new_position:
            return None
    
    moves.append(new_move)

generate_moves_sig = MoveDescriptionTuple.lsttype(i1MatrixContType, T1)
@cc.export("generate_moves", generate_moves_sig)
@nb.njit(generate_moves_sig)
def generate_moves(blocks, np_active_piece):
    moves = nb.typed.List.empty_list(MoveDescriptionTuple.type)
    if np_active_piece is None:
        return moves

    
    row_layers = BoardParsing.parse_row_layers(blocks)
    col_layers = BoardParsing.parse_row_layers(blocks.T)
    peaks = (23-2) - BoardParsing.find_block_peaks(col_layers)
    connected_sets = BoardParsing.parse_connected_sets(col_layers, row_layers)
    overhangs = MoveFiltering.find_overhangs(connected_sets, col_layers)

    varient_set = PieceLUT.get_piece_set(np_active_piece.name.item())
    for orientation_index, each_varient in enumerate(varient_set):
        h, w = each_varient.shape
        for x in range(10):
            np_piece = np_active_piece.copy()
            target_y = np.min(peaks[x:x+w])
            NumbaTetromino.set_tetromino_position(np_piece, (x, target_y))
            NumbaTetromino.set_tetromino_orientation(np_piece, orientation_index)
            move = drop_piece(blocks, np_piece, each_varient)
            if move is not None:
                moves.append(move)

        destination_piece = np_active_piece.copy()
        NumbaTetromino.set_tetromino_orientation(destination_piece, orientation_index)

        for overhang in overhangs:
            ox, oy = overhang.origin
            
            for x in range(ox-w+1, ox+1):
                NumbaTetromino.set_tetromino_position(destination_piece, (x, oy))
                move = tuck_piece(blocks, col_layers, row_layers, overhang, destination_piece, np_active_piece)

                if move is not None:
                    append_unique_move(moves, move)
    
    return moves

if __name__ == "__main__":
    cc.compile()
