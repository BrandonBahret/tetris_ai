import numba as nb

from Helpers import ArrayFuncs
from Types.NumbaDefinitions import IntSetList, IntSet

@nb.njit(IntSet())
def empty_set():
    aset = {int(-1)}
    aset.clear()
    return aset

@nb.njit([IntSet(IntSetList, nb.int64), IntSet(IntSetList, nb.int32)])
def find_set_with_element(set_list, element):
    for each_set in set_list:
        if element in each_set:
            return each_set
    return empty_set()

@nb.njit(IntSetList(IntSetList, IntSet))
def find_jointed_sets(set_list, target_set):
    matches = nb.typed.List.empty_list(IntSet)
    for each_set in set_list:
        if not each_set.isdisjoint(target_set):
            matches.append(each_set)
    return matches

@nb.njit(IntSet(IntSetList, nb.int64))
def find_set_closest_to(set_list, target_element):
    measure_proximity = lambda input_set, target: abs(target - ArrayFuncs.sum_array(input_set)/len(input_set))
    measured_sets = [(measure_proximity(each, target_element), each) for each in set_list]
    measured_sets.sort(key=lambda e: e[0])
    return measured_sets[0][1]
