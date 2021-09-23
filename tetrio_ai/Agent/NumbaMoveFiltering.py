import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import numpy as np
import numba as nb
from numba.pycc import CC

from typing import NamedTuple

from Agent import NumbaBoardParsing as BoardParsing

from Helpers import ArrayFuncs, SetFuncs
from Types.NumbaDefinitions import NbExpose
from Types.NumbaDefinitions import i1MatrixContType
from Types.NumbaDefinitions import T1
from Types.NumbaDefinitions import ConnectionLayer, ConnectionGap
from Types.NumbaDefinitions import ConnectedSet
from Types.NumbaDefinitions import OverhangSet
from Types.NumbaDefinitions import MoveDescriptionTuple
from Types.NumbaDefinitions import MetricBundle
from Types.NumbaDefinitions import ModelWeights


cc = CC('MoveFiltering')
# Uncomment the following line to print out the compilation steps
# cc.verbose = True

### An overhang in tetris will be defined as any vacant-space underneath a block iff it connects to a path to the surface
def print_connection_gap(gap):
    print(f"layer={gap.layer_index:0>2d}, gap={gap.gap_set}")

def print_connected_layer(layer):
    print(f"layer={layer.layer_index:0>2d}, gaps={layer.gaps_list}")

def print_connected_set(connected_set):
    if connected_set is None:
        print(None)
        return

    print(f"is enclosed = {connected_set.is_enclosed}")
    for layer in connected_set.layers:
        print_connected_layer(layer)


close_holes_sig = i1MatrixContType(ConnectionLayer.lsttype)
@cc.export("close_holes_and_overhangs", close_holes_sig)
@nb.njit(close_holes_sig, cache=True)
def close_holes_and_overhangs(col_layers):
    closed_board = np.zeros((23, 10), dtype=np.int8)

    peaks = BoardParsing.find_block_peaks(col_layers)
    for col_layer in col_layers:
        x = col_layer.layer_index
        y = 22 - peaks[x]

        if y != 22:
            closed_board[y:, x] = 4

    return closed_board

well_height_sig = nb.int64(ConnectionGap.type, ConnectionLayer.lsttype)
@cc.export("measure_well_height", well_height_sig)
@nb.njit(well_height_sig, cache=True)
def measure_well_height(well_mouth, col_layers):
    max_well_height = 0
    well_mouth_y = well_mouth.layer_index

    peaks = BoardParsing.find_block_peaks(col_layers)
    for x in well_mouth.gap_set:
        well_floor_y = 22 - peaks[x]
        well_height = well_floor_y - well_mouth_y +1
        
        if well_height > max_well_height:
            max_well_height = well_height

    return max_well_height

well_area_sig = nb.int64(ConnectionGap.type, ConnectionLayer.lsttype)
@cc.export("measure_well_area", well_area_sig)
@nb.njit(well_area_sig, cache=True)
def measure_well_area(well_mouth, col_layers):
    aggregate_area = 0
    well_mouth_y = well_mouth.layer_index

    peaks = BoardParsing.find_block_peaks(col_layers)
    for x in well_mouth.gap_set:
        well_floor_y = 22 - peaks[x]
        aggregate_area += well_floor_y - well_mouth_y +1
    
    return aggregate_area

max_well_height_sig = nb.int64(ConnectionGap.lsttype, ConnectionLayer.lsttype)
@cc.export("find_max_well_height", max_well_height_sig)
@nb.njit(max_well_height_sig, cache=True)
def find_max_well_height(wells, col_layers):
    max_well_height = 0

    for well_mouth in wells:
        well_height = measure_well_height(well_mouth, col_layers)
        if well_height > max_well_height:
            max_well_height = well_height
    
    return max_well_height

well_mouth_sig = nb.boolean(ConnectionGap.type, ConnectionLayer.lsttype, ConnectionLayer.lsttype)
@cc.export("is_well_mouth", well_mouth_sig)
@nb.njit(well_mouth_sig, cache=True)
def is_well_mouth(connection_gap, row_layers, col_layers):
    gap = connection_gap.gap_set
    y = connection_gap.layer_index

    max_well_width = 3
    if gap == SetFuncs.empty_set() or len(gap) > max_well_width:
        return False

    if y+1 <= 22:
        jointed_sets = SetFuncs.find_jointed_sets(row_layers[y+1].gaps_list, gap)
        if len(jointed_sets) == 0:
            return False
        
        if len(gap) >= len(jointed_sets[0]):
            if y-1 >= 0:
                prev_jointed_sets = SetFuncs.find_jointed_sets(row_layers[y-1].gaps_list, gap)
                if len(prev_jointed_sets) > 0 and len(prev_jointed_sets[0]) <= max_well_width:
                    return False

    if measure_well_height(connection_gap, col_layers) < 3:
        return False

    return True


well_mouths_sig = ConnectionGap.lsttype(ConnectionLayer.lsttype, ConnectionLayer.lsttype)
@cc.export("find_well_mouths", well_mouths_sig)
@nb.njit(well_mouths_sig, cache=True)
def find_well_mouths(row_layers, col_layers):
    mouths = nb.typed.List.empty_list(ConnectionGap.type)

    well_memo = close_holes_and_overhangs(col_layers)
    well_memo_row_layers = BoardParsing.parse_row_layers(well_memo)

    for layer in well_memo_row_layers:
        if len(layer.gaps_list[0]) == 10:
            continue

        y = layer.layer_index
        for gap in layer.gaps_list:
            potential_mouth = ConnectionGap.new(y, gap)

            if is_well_mouth(potential_mouth, row_layers, col_layers):
                jointed_gaps = BoardParsing.find_jointed_gaps(mouths, potential_mouth.gap_set)
                if len(jointed_gaps) != 0:
                    continue
            
                mouths.append(potential_mouth)

    return mouths

connected_area_sig = nb.int64(ConnectedSet.type)
@cc.export("measure_connected_area", connected_area_sig)
@nb.njit(connected_area_sig, cache=True)
def measure_connected_area(connected_set):
    area = 0
    for layer in connected_set.layers:
        for gap in layer.gaps_list:
            area += len(gap)

    return area


find_holes_sig = ConnectedSet.lsttype(ConnectedSet.lsttype)
@cc.export("find_holes", find_holes_sig)
@nb.njit(find_holes_sig, cache=True)
def find_holes(connected_sets):
    holes = nb.typed.List.empty_list(ConnectedSet.type)
    for each_connected_set in connected_sets:
        if each_connected_set.is_enclosed:
            holes.append(each_connected_set)

    return holes

find_overhangs_sig = OverhangSet.lsttype(ConnectedSet.lsttype, ConnectionLayer.lsttype)
@cc.export("find_overhangs", find_overhangs_sig)
@nb.njit(find_overhangs_sig)
def find_overhangs(connected_sets, column_layers):
    crop_connected_set_at_row = lambda cs, row_i: ConnectedSet.new(cs.is_enclosed, cs.layers[cs.layers[0].layer_index-row_i:])

    overhangs = nb.typed.List.empty_list(OverhangSet.type)

    for col_layer in column_layers:
        for col_gap in col_layer.gaps_list[1:]:
            row_i = max(col_gap)
            col_i = col_layer.layer_index
            cs = BoardParsing.find_connected_set(connected_sets, col_i, row_i)

            if cs is not None and not cs.is_enclosed:
                cs = crop_connected_set_at_row(cs, row_i)
                overhang_mouth = BoardParsing.find_connected_set_mouth(cs, (col_i, row_i), column_layers)
                each_overhang = OverhangSet.new((col_i, row_i), cs, overhang_mouth)
                overhangs.append(each_overhang)
    
    return overhangs

widths_of_overhang_mouths_sig = nb.int64(OverhangSet.lsttype)
@cc.export("sum_widths_of_overhang_mouths", widths_of_overhang_mouths_sig)
@nb.njit(widths_of_overhang_mouths_sig, cache=True)
def sum_widths_of_overhang_mouths(overhangs):
    total = 0
    for each in overhangs:
        total += len(each.mouth.gap_set)

    return total

layer_connectivity_sig = nb.float64(ConnectionLayer.lsttype)
@cc.export("measure_layer_connectivity", layer_connectivity_sig)
@nb.njit(layer_connectivity_sig)
def measure_layer_connectivity(row_layers):
    connectivity_score = 0

    for layer in row_layers:
        connectivity = 10
        for gap in layer.gaps_list:
            connectivity -= len(gap)

        connectivity /= len(layer.gaps_list)
        connectivity_score += connectivity

    return connectivity_score / (9.0 * 23)

parse_metrics_sig = MetricBundle.type(i1MatrixContType)
@cc.export("parse_metrics", parse_metrics_sig)
@nb.njit(parse_metrics_sig)
def parse_metrics(blocks):
    row_layers = BoardParsing.parse_row_layers(blocks)
    col_layers = BoardParsing.parse_row_layers(blocks.T)
    connected_sets = BoardParsing.parse_connected_sets(col_layers, row_layers)
    layers_data = (row_layers, col_layers, connected_sets)
    
    holes = find_holes(connected_sets)
    overhangs = find_overhangs(connected_sets, col_layers)
    well_mouths = find_well_mouths(row_layers, col_layers)
    features = (holes, overhangs, well_mouths)


    hole_count = len(holes)
    overhang_count = len(overhangs)
    well_count = len(well_mouths)
    feature_lengths = (hole_count, overhang_count, well_count)

    agg_well_height = 0 
    for well in well_mouths:
        agg_well_height += measure_well_height(well, col_layers)

    hole_area_sum = ArrayFuncs.sum_array(np.asarray([measure_connected_area(h) for h in holes]))
    overhang_mouth_sum = sum_widths_of_overhang_mouths(overhangs)
    max_well_height = find_max_well_height(well_mouths, col_layers)
    row_connectivity = measure_layer_connectivity(row_layers)
    metrics = (hole_area_sum, overhang_mouth_sum, max_well_height, agg_well_height, row_connectivity)

    return MetricBundle.new(*layers_data, *features, *feature_lengths, *metrics)

# @nb.njit(nb.int64(T1))
# def get_weight_index(piece):
#     piece_names = str("JLSZITO")
#     piece_name = str(piece.name.item())
#     return piece_names.index(piece_name)

weight_index_sig = nb.int64(T1)
@cc.export("get_weight_index", weight_index_sig)
@nb.njit(weight_index_sig)
def get_weight_index(piece):
    piece_name = str(piece.name.item())
    piece_classes = [str("SZ"), str("JL"), str("I"), str("O"), str("T")]

    i = 0
    for i, p_class in enumerate(piece_classes):
        if piece_name in p_class:
            break

    return i

prescore_sig = nb.int64(MoveDescriptionTuple.type, MetricBundle.type, ModelWeights.type)
@cc.export("prescore", prescore_sig)
@nb.njit(prescore_sig)
def prescore(move, initial_bundle, weights):
    piece = move.destinations[0]
    weight_index = get_weight_index(piece)

    blocks = move.target_outcome
    bundle = parse_metrics(blocks)

    # score =  ((bundle.agg_well_height - initial_bundle.agg_well_height) * weights.agg_well_height[weight_index].value)
    # score += ((bundle.hole_count - initial_bundle.hole_count) * weights.hole_count[weight_index].value)
    # score += ((bundle.overhang_count - initial_bundle.overhang_count) * weights.overhang_count[weight_index].value)
    # score += ((bundle.well_count - initial_bundle.well_count) * weights.well_count[weight_index].value)
    # score += ((bundle.hole_area_sum - initial_bundle.hole_area_sum) * weights.hole_area_sum[weight_index].value)
    # score += ((bundle.overhang_mouth_sum - initial_bundle.overhang_mouth_sum) * weights.overhang_mouth_sum[weight_index].value)
    # score += ((bundle.max_well_height - initial_bundle.max_well_height) * weights.max_well_height[weight_index].value)

    # if hole_count > 0:
    #     score =  (bundle.agg_well_height - initial_bundle.agg_well_height)
    #     score += (bundle.hole_count - initial_bundle.hole_count)
    #     score += ((bundle.overhang_count - initial_bundle.overhang_count) * weights.c0overhang_count[weight_index].value)
    #     score += (bundle.well_count - initial_bundle.well_count)
    #     score += (bundle.hole_area_sum - initial_bundle.hole_area_sum)
    #     score += (bundle.overhang_mouth_sum - initial_bundle.overhang_mouth_sum)
    #     score += (bundle.max_well_height - initial_bundle.max_well_height)
    #     score += (bundle.row_connectivity - initial_bundle.row_connectivity)

    # else:
    score =  (bundle.agg_well_height - initial_bundle.agg_well_height)
    score += (bundle.hole_count - initial_bundle.hole_count)
    score += (bundle.overhang_count - initial_bundle.overhang_count)
    score += (bundle.well_count - initial_bundle.well_count)
    score += (bundle.hole_area_sum - initial_bundle.hole_area_sum)
    score += (bundle.overhang_mouth_sum - initial_bundle.overhang_mouth_sum)
    score += (bundle.max_well_height - initial_bundle.max_well_height)
    score += (bundle.row_connectivity - initial_bundle.row_connectivity)


    return score


filter_moves_sig = MoveDescriptionTuple.lsttype(MoveDescriptionTuple.lsttype, MetricBundle.type, nb.int64, ModelWeights.type)
@cc.export("filter_moves", filter_moves_sig)
@nb.njit(filter_moves_sig)
def filter_moves(moves, initial_bundle, target_length, weights):
    prescored_moves = [(prescore(m, initial_bundle, weights), m) for m in moves]
    prescored_moves.sort(key = lambda e: e[0])

    filtered_moves = nb.typed.List.empty_list(MoveDescriptionTuple.type)
    for each_pair in prescored_moves[:target_length]:
        filtered_moves.append(each_pair[1])

    return filtered_moves

if __name__ == "__main__":
    cc.compile()
