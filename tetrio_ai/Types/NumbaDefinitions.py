import numpy as np
import numba as nb

from typing import NamedTuple
from collections import namedtuple

nb_list = nb.typed.typedlist.List

def NbExpose(named_tuple):
    class TupleType(NamedTuple):
        new: NamedTuple
        type: nb.types.TypeRef
        lsttype: nb.types.TypeRef

    field_names = named_tuple._fields
    field_types = tuple(named_tuple._field_types.values())

    tuple_name = named_tuple.__qualname__
    mangled_tuple_name = f"__{tuple_name}"
    __named_tuple = namedtuple(mangled_tuple_name, field_names)
    __named_tuple_type = nb.types.NamedTuple.from_types(field_types, __named_tuple)
    __named_tuple_lsttype = nb.types.ListType(__named_tuple_type)

    globals()[mangled_tuple_name] = __named_tuple

    def unwrap_iterable(iterable, flatten, compact, depth):
        end_char = "\n" if not compact else ""
        tab_char = "  " if not compact else ""
        list_len = len(iterable)

        repr_str = f"[{end_char}"
        for i, each_member in enumerate(iterable):
            if isinstance(each_member, (nb_list, list)):
                member_rep = unwrap_iterable(each_member, flatten, compact, depth+1)
            elif hasattr(each_member, "value"):
                member_rep = str(each_member.value)
            elif hasattr(each_member, "repr"):
                member_rep = each_member.str(flatten, compact, depth+1)
            else:
                member_rep = str(each_member)

            if i < list_len-1:
                member_rep += f", {end_char}"

            if not compact:
                repr_str += f"{tab_char*(depth+1)}{member_rep}"
            else:
                repr_str += member_rep

        repr_str += f"{end_char}{tab_char*(depth)}]"
        return repr_str


    def str_fn(tuple_type, flatten=False, compact=True, depth=0):
        field_count = len(tuple_type._fields)
        flatten |= (depth >= 2)
        end_char = "\n" if not flatten else ""
        tab_char = "  " if not flatten else ""

        repr_str = f"{tuple_name}({end_char}"
        for i, field in enumerate(tuple_type._fields):
            each_member = tuple_type[i]
            if isinstance(each_member, (nb_list, list)):
                member_rep = unwrap_iterable(each_member, flatten, compact, depth+1)
            elif hasattr(each_member, "value"):
                member_rep = str(each_member.value)
            elif hasattr(each_member, "repr"):
                member_rep = each_member.str(flatten, compact, depth=depth+1)
            else:
                member_rep = str(each_member)

            repr_str += f"{tab_char*(depth+1)}{field} = {member_rep}"
            if i < field_count-1:
                repr_str += f", {end_char}"
        
        repr_str += f"{end_char}{tab_char*depth})"
        
        return repr_str

    def repr_fn(tuple_type, flatten=False, compact=True):
        print(tuple_type.str(flatten, compact))

    __named_tuple.str = str_fn
    __named_tuple.repr = repr_fn

    return TupleType(__named_tuple, __named_tuple_type, __named_tuple_lsttype)


## Simple types
# Integer types
IntPair = nb.types.UniTuple(nb.int64, 2)

IntArray = nb.types.Array(nb.int64, 1, 'C')
IntArrayList = nb.types.ListType(IntArray)
Int32Array = nb.types.Array(nb.int32, 1, 'C')
Int8Array = nb.types.Array(nb.int8, 1, 'C')

IntSet = nb.types.Set(nb.int64)
IntSetList = nb.types.ListType(IntSet)
IntSet2dList = nb.types.ListType(IntSetList)

# Boolean types
BooleanArray = nb.types.Array(nb.boolean, 1, 'C')
BooleanList = nb.types.ListType(nb.boolean)

# Char types
np_unichar = np.dtype('<U1')
UnicharType = nb.from_dtype(np_unichar)
UnicodeListType = nb.types.ListType(nb.types.unicode_type)

## Common compund types
# Matrix Definitions
i1MatrixType = nb.types.Array(nb.int8, 2, 'A')

i1MatrixContType = nb.types.Array(nb.int8, 2, 'C')
i1MatrixFortType = nb.types.Array(nb.int8, 2, 'F')
i1MatrixAnyType = nb.types.Array(nb.int8, 2, 'A')
i1MatrixTypes = [i1MatrixContType, i1MatrixFortType, i1MatrixAnyType]

i1MatrixListType = nb.types.ListType(i1MatrixType)
i1MatrixDictType = nb.types.DictType(UnicharType, i1MatrixType)
i1MatrixListDictType = nb.types.DictType(UnicharType, i1MatrixListType)

# Tetromino Definitions
np_coord = np.dtype([('x', np.int8), ('y', np.int8)])
CoordType = nb.from_dtype(np_coord)

np_tetromino = np.dtype({
    'names': ['name', 'orientation', 'position', 'is_held_piece'],
    'formats': [np_unichar, np.int8, np_coord, np.bool]
})
TetrominoType = nb.from_dtype(np_tetromino)
T1 = nb.types.Array(TetrominoType, 1, 'C')

# MoveDescription Definitions
@NbExpose
class MoveDescriptionTuple(NamedTuple):
    destinations: nb.types.ListType(T1)
    target_placement: i1MatrixContType
    target_outcome: i1MatrixContType
    target_placement_simple: i1MatrixContType

# ConnectedSet Definitions
@NbExpose
class ConnectionGap(NamedTuple):
    layer_index: nb.int64
    gap_set: IntSet

@NbExpose
class ConnectionLayer(NamedTuple):
    layer_index: nb.int64
    gaps_list: IntSetList

@NbExpose
class ConnectedSet(NamedTuple):
    is_enclosed: nb.boolean
    layers: ConnectionLayer.lsttype

@NbExpose
class OverhangSet(NamedTuple):
    origin: IntPair
    connected_set: ConnectedSet.type
    mouth: ConnectionGap.type

@NbExpose
class GeneticWeightBundle(NamedTuple):
    minimum: nb.float64
    maximum: nb.float64
    value: nb.float64

# Result definitions
@NbExpose
class MetricBundle(NamedTuple):
    row_layers: ConnectionLayer.lsttype
    col_layers: ConnectionLayer.lsttype
    connected_sets: ConnectedSet.lsttype

    holes: ConnectedSet.lsttype
    overhangs: OverhangSet.lsttype
    wells: ConnectionGap.lsttype

    hole_count: nb.int64
    overhang_count: nb.int64
    well_count: nb.int64

    hole_area_sum: nb.int64
    overhang_mouth_sum: nb.int64
    max_well_height: nb.int64
    agg_well_height: nb.int64
    row_connectivity: nb.float64

@NbExpose
class MoveScorePair(NamedTuple):
    move: MoveDescriptionTuple.type
    score: nb.float64

@NbExpose
class PredictionResult(NamedTuple):
    best_move: MoveScorePair.type
    move_list: MoveDescriptionTuple.lsttype

# @NbExpose
# class ModelWeights(NamedTuple):
#     agg_well_height    : GeneticWeightBundle.lsttype
#     hole_count         : GeneticWeightBundle.lsttype
#     well_count         : GeneticWeightBundle.lsttype
#     overhang_count     : GeneticWeightBundle.lsttype
#     hole_area_sum      : GeneticWeightBundle.lsttype
#     overhang_mouth_sum : GeneticWeightBundle.lsttype
#     max_well_height    : GeneticWeightBundle.lsttype

#     line_total         : GeneticWeightBundle.lsttype
#     flatness           : GeneticWeightBundle.lsttype
#     row_connectivity   : GeneticWeightBundle.lsttype
#     aggregate_height   : GeneticWeightBundle.lsttype

@NbExpose
class ModelWeights(NamedTuple):
    # agg_well_height    : GeneticWeightBundle.lsttype
    # c0hole_count         : GeneticWeightBundle.lsttype
    # c1hole_count         : GeneticWeightBundle.lsttype
    # well_count         : GeneticWeightBundle.lsttype
    # c0overhang_count     : GeneticWeightBundle.lsttype
    # c1overhang_count     : GeneticWeightBundle.lsttype
    # hole_area_sum      : GeneticWeightBundle.lsttype
    # overhang_mouth_sum : GeneticWeightBundle.lsttype
    # max_well_height    : GeneticWeightBundle.lsttype

    # line_total         : GeneticWeightBundle.lsttype
    # flatness           : GeneticWeightBundle.lsttype
    c0row_connectivity   : GeneticWeightBundle.lsttype
    c1row_connectivity   : GeneticWeightBundle.lsttype
    # aggregate_height   : GeneticWeightBundle.lsttype
