# import os, sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
# sys.path.append(parentdir)

from numba.pycc import CC
import numba as nb
import numpy as np

from Types.NumbaDefinitions import UnicharType
from Types.NumbaDefinitions import i1MatrixType, i1MatrixListType, i1MatrixListDictType

cc = CC('PieceLUT_AOT')
# Uncomment the following line to print out the compilation steps
# cc.verbose = True

@nb.experimental.jitclass
class PieceLUT(object):
    pieces: i1MatrixListDictType

    def __init__(self):
        self.pieces = nb.typed.Dict.empty(
            key_type = UnicharType,
            value_type = i1MatrixListType
        )

        T = np.array([
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.int8)

        J = np.array([
            [1, 1, 1],
            [0, 0, 1],
        ], dtype=np.int8)

        L = np.array([
            [1, 1, 1],
            [1, 0, 0],
        ], dtype=np.int8)

        S = np.array([
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.int8)

        Z = np.array([
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=np.int8)

        I = np.array([
            [1, 1, 1, 1],
        ], dtype=np.int8)

        O = np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.int8)

        self.pieces["T"] = self.list_like([self.rot90(T, i) for i in range(4)], i1MatrixType)
        self.pieces["J"] = self.list_like([self.rot90(J, i) for i in range(4)], i1MatrixType)
        self.pieces["L"] = self.list_like([self.rot90(L, i) for i in range(4)], i1MatrixType)
        self.pieces["S"] = self.list_like([self.rot90(S, i) for i in range(2)], i1MatrixType)
        self.pieces["Z"] = self.list_like([self.rot90(Z, i) for i in range(2)], i1MatrixType)
        self.pieces["I"] = self.list_like([self.rot90(I, i) for i in range(2)], i1MatrixType)
        self.pieces["O"] = self.list_like([O], i1MatrixType)

    def list_like(self, lst, element_type):
        new_list = nb.typed.List.empty_list(element_type)
        for each_element in lst:
            new_list.append(each_element)
        return new_list
    
    def rot90(self, array, k=0):
        k %= 4

        if k == 0:
            return array
        elif k == 1:
            return np.fliplr(array).T
        elif k == 2:
            return np.flip(array)
        else:
            return np.flipud(array).T

@cc.export("get_piece_set", i1MatrixListType(UnicharType))
@nb.njit(i1MatrixListType(UnicharType), cache=True)
def get_piece_set(name):
    return PieceLUT().pieces[name]

if __name__ == "__main__":
    cc.compile()
