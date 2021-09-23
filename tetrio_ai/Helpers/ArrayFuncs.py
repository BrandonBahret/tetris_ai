import numpy as np
import numba as nb

from Types.NumbaDefinitions import BooleanArray
from Types.NumbaDefinitions import IntArray, IntSet, Int32Array

__all__ = [
    "isclose",
    "index_list",
    "find_unique"
]

@nb.njit(nb.float32(nb.float32, nb.float32))
def isclose(a, b):
    '''determines if one float is close to another.'''
    atol = 0.0
    rtol = 1e-04
    
    return abs(a - b) <= (atol + rtol * abs(b))

@nb.njit(nb.int64(nb.types.ListType(nb.float32), nb.float32))
def index_list(lst, value):
    '''finds the index of the value in the list. return -1 if not found.'''
    for index, each_value in enumerate(lst):
        if isclose(each_value, value):
            return index
    
    return -1

@nb.njit(nb.int8[:](nb.int8[:]))
def find_unique(arr):
    '''find the indexes of unique values in a one-dimensional array.'''
    unique_indexs = [i for (i, e) in enumerate(arr) if i == 0 or arr[i-1] != e]
    return np.array(unique_indexs, dtype=np.int8)

@nb.njit([nb.int64(each_type) for each_type in [IntArray, Int32Array, IntSet]])
def sum_array(array):
    total = 0
    for element in array:
        total += element
    
    return total

@nb.njit([nb.boolean(IntArray), nb.boolean(BooleanArray)])
def any_true(array):
    for element in array:
        if element is True:
            return True
    
    return False

@nb.njit([nb.boolean(IntArray), nb.boolean(BooleanArray)])
def all_true(array):
    for element in array:
        if element is False:
            return False
    
    return True
