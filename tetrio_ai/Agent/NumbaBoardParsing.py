import numba as nb
import numpy as np

from Helpers import ArrayFuncs, SetFuncs

from Types.NumbaDefinitions import IntSetList, IntSet, IntArray, IntPair
from Types.NumbaDefinitions import i1MatrixContType, i1MatrixTypes
from Types.NumbaDefinitions import ConnectionLayer, ConnectionGap
from Types.NumbaDefinitions import ConnectedSet


@nb.njit(IntSetList(IntArray), cache=True)
def split_consecutive_sequences(array):
    arithmetic_diff = array[1:] - array[:-1]
    consecutive_indices = np.where(arithmetic_diff != 1)[0]+1
    consecutive_sequences = np.split(array, consecutive_indices)

    return nb.typed.List([set(seq) for seq in consecutive_sequences])

@nb.njit([ConnectionLayer.lsttype(each_type) for each_type in i1MatrixTypes], cache=True)
def parse_row_layers(blocks):    
    row_layers = nb.typed.List.empty_list(ConnectionLayer.type)
    for row_i, row in enumerate(blocks):
        vacant_list = np.where(row == 0)[0]
        gaps_list = split_consecutive_sequences(vacant_list)
        row_layers.append(ConnectionLayer.new(row_i, gaps_list))
    
    return row_layers

@nb.njit(nb.boolean(ConnectionLayer.type, ConnectionLayer.lsttype), cache=True)
def is_layer_enclosed(row_layer, col_layers):
    row_i = row_layer.layer_index
    for xs in row_layer.gaps_list:
        for x in xs:
            surface_gap = col_layers[x].gaps_list[0]
            if row_i not in surface_gap:
                return True
    
    return False

@nb.njit(ConnectedSet.type(ConnectionLayer.lsttype, ConnectionLayer.lsttype, nb.int64, nb.int64), cache=True)
def find_all_connections(row_layers, col_layers, x, y):
    starting_set = SetFuncs.find_set_with_element(row_layers[y].gaps_list, x)
    starting_sets = nb.typed.List.empty_list(IntSet)
    starting_sets.append(starting_set)

    connected_layers = nb.typed.List.empty_list(ConnectionLayer.type)
    last_appended = ConnectionLayer.new(y, starting_sets)
    connected_layers.append(last_appended)

    for row_i in range(y-1, -1, -1):
        this_layer = row_layers[row_i]
        is_final_layer = (not is_layer_enclosed(this_layer, col_layers))
        new_connections = nb.typed.List.empty_list(IntSet)

        for each_gap in last_appended.gaps_list:
            connections = SetFuncs.find_jointed_sets(this_layer.gaps_list, each_gap)
            # connections = [c for c in connections if c not in new_connections]
            new_connections.extend(connections)
            is_enclosed = len(new_connections) == 0

            if is_final_layer or is_enclosed:
                if not is_enclosed:
                    connected_layers.append(ConnectionLayer.new(row_i, new_connections))
                return ConnectedSet.new(is_enclosed, connected_layers)

        last_appended = ConnectionLayer.new(row_i, new_connections)
        connected_layers.append(last_appended)

    return ConnectedSet.new(False, connected_layers)

@nb.njit(nb.boolean(ConnectedSet.type, nb.int64, nb.int64), cache=True)
def is_point_contained(connected_set, x, y):
    for row in connected_set.layers:
        if row.layer_index != y:
            continue

        for gap in row.gaps_list:
            if x in gap:
                return True

    return False

@nb.njit(ConnectedSet.lsttype(ConnectionLayer.lsttype, ConnectionLayer.lsttype), cache=True)
def parse_connected_sets(col_layers, row_layers):
    connected_sets = nb.typed.List.empty_list(ConnectedSet.type)

    for col_layer in col_layers:
        for col_gap in col_layer.gaps_list[1:]:
            row_i = int(max(col_gap))
            col_i = int(col_layer.layer_index)

            if ArrayFuncs.any_true(np.asarray([is_point_contained(cs, col_i, row_i) for cs in connected_sets])):
                continue

            each_connected_set = find_all_connections(row_layers, col_layers, col_i, row_i)
            connected_sets.append(each_connected_set)
    
    return connected_sets

@nb.njit(nb.types.Optional(ConnectedSet.type)(ConnectedSet.lsttype, nb.int64, nb.int64), cache=True)
def find_connected_set(connected_sets, x, y):
    for cs in connected_sets:
        if is_point_contained(cs, x, y):
            return cs
    
    return None

@nb.njit(ConnectionGap.lsttype(ConnectionGap.lsttype, IntSet), cache=True)
def find_jointed_gaps(gap_list, target_set):
    matches = nb.typed.List.empty_list(ConnectionGap.type)
    for each_gap in gap_list:
        if not each_gap.gap_set.isdisjoint(target_set):
            matches.append(each_gap)
    
    return matches

@nb.njit(ConnectionGap.type(ConnectedSet.type, IntPair, ConnectionLayer.lsttype))
def find_connected_set_mouth(connected_set, origin, col_layers):
    row_full_set = set(nb.prange(10))

    for i, layer in enumerate(connected_set.layers):
        layer_i = layer.layer_index
        
        if not is_layer_enclosed(layer, col_layers):
            xs_set = SetFuncs.find_set_closest_to(layer.gaps_list, origin[0])
            
            if i-1 >= 0:
                previous_layer = connected_set.layers[i-1]
                previous_xs = SetFuncs.find_jointed_sets(previous_layer.gaps_list, xs_set)[0]
                return ConnectionGap.new(layer_i, xs_set - (row_full_set - previous_xs) )
            else:
                return ConnectionGap.new(layer_i, xs_set)
            
    return ConnectionGap.new(-1, SetFuncs.empty_set())

@nb.njit(IntArray(ConnectionLayer.lsttype), cache=True)
def find_block_peaks(col_layers):
    highest_block_ys = np.full((10), 22, dtype=np.int64)

    for col_layer in col_layers:
        x = col_layer.layer_index
        topmost_col_gap = list(col_layer.gaps_list[0])

        if len(topmost_col_gap) > 0:
            highest_block_ys[x] = max(0, 22 - (topmost_col_gap[-1]+1))
            
    return highest_block_ys
