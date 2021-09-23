import matplotlib.pyplot as plt
from skimage import io
import numba as nb
import numpy as np
from frame_display import FrameDisplay

import pickle
import time
import math

from Agent import NumbaMovePrediction, MovePrediction
from Agent import NumbaBoardParsing
from Agent import NumbaTetromino
from Agent import PieceLUT

from FrameGrabber.Frame import Frame

from typing import NamedTuple
from Types.NumbaDefinitions import NbExpose
from Types.NumbaDefinitions import T1
from Types.NumbaDefinitions import i1MatrixContType, i1MatrixAnyType
from Types.NumbaDefinitions import MoveDescriptionTuple

cmap = {
    0 : (97, 18, 117),
    1 : (214, 214, 43),
    2 : (59, 189, 8),
    3 : (201, 57, 38),
    4 : (38, 201, 185),
    5 : (60, 60, 255),
}

def map_colors(blocks, color_map):
    img = blocks.astype(np.int64).copy()
    img = Frame.change_nchannels(img, 3)
    mapped_img = img.copy()
    for key, color in color_map.items():
        mapped_img[mapped_img[:,:,0] == key] = color
    mapped_img = Frame.resize(mapped_img, 500)

    return mapped_img

def show_blocks(blocks, disp):
    blocks_im = map_colors(blocks, cmap)
    disp.update(blocks_im) 


np_unichar = np.dtype('<U1')
UnicharType = nb.from_dtype(np_unichar)
UnicharListType = nb.types.ListType(UnicharType)

def load_inputs():
    with open("resources//simulation_inputs.pyobj", "rb+") as file:
        sim_inputs = nb.typed.List(pickle.load(file))
    
    return sim_inputs

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

@nb.njit
def square_wave(x, pulse_gap=1, pulse_width=1):
    if pulse_gap == 0:
        return pulse_width != 0

    cycle = pulse_width + pulse_gap
    x = x % cycle

    n = (x % pulse_gap) + 1
    d = x + 1
    quotient = n / d
    
    y = -math.floor(quotient) +1
    return y > 0

@NbExpose
class SimulationBundle(NamedTuple):
    gameover: nb.boolean
    hold_count: nb.int64
    line_count: nb.int64
    final_board_height: nb.int64
    tetris_count: nb.int64
    iter_count: nb.int64
    max_iters: nb.int64

def py_simulate(blocks_in, agent, iterations, delay=0):
    blocks = blocks_in.copy()
    disp = FrameDisplay(map_colors(blocks, cmap))

    weights = agent.weights_as_bundles

    gameover = False
    line_count = 0
    tetris_count = 0
    hold_count = 0

    bag = SevenBag(load_inputs())
    held_piece = None
    is_held_disabled = False

    for i in range(iterations):
        if is_gameover(blocks):
            gameover = True
            break

        active_piece = bag.get_next_piece()
        piece_blocks = get_blocks(active_piece)
        next_queue = nb.typed.List(bag.next_queue)

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
                active_piece['is_held_piece'].fill(False)
                target_move = NumbaMovePrediction.predict_move(blocks, active_piece, next_queue, held_piece, is_held_disabled, weights)

        else:
            is_held_disabled = False

        blocks = target_move.target_outcome
        lines = MovePrediction.count_lines(target_move.target_placement_simple)
        line_count += lines
        tetris_count += lines//4

        if square_wave(i, 0, 1):
            # clear_output(wait=True)
            show_blocks(target_move.target_placement, disp)

            if lines > 0:
                # clear_output(wait=True)
                time.sleep(int(delay*0.2))
                show_blocks(blocks, disp)
                time.sleep(int(delay*0.2))

        if delay > 0:
            time.sleep(delay)

    col_layers = NumbaBoardParsing.parse_row_layers(blocks.T)
    final_board_height = MovePrediction.measure_aggregate_block_height(col_layers)
    disp.close()
    metrics = SimulationBundle.new(gameover, hold_count, line_count, final_board_height, tetris_count, i+1, iterations)
    return metrics


from Types.NumbaDefinitions import ModelWeights, GeneticWeightBundle

class GeneticWeight:
    def __init__(self, minimum, maximum, step=0.01):
        self.__value_range = (float(minimum), float(maximum))
        self.__value_space = np.round(np.arange(minimum, maximum+step, step), decimals=4)
        self.__value_space_size = len(self.__value_space)
        self.__value = np.random.choice(self.__value_space)
        # self.__value = (maximum - minimum) * np.random.sample() + minimum

    def __repr__(self):
        return self.to_bundle().str(flatten=True)

    def to_bundle(self):
        minimum, maximum = self.__value_range
        value = self.__value
        return GeneticWeightBundle.new(minimum, maximum, value)

    @property
    def value_space(self):
        return self.__value_space

    @property
    def value_range(self):
        return self.__value_range

    @property
    def value(self):
        return self.__value

    def set_weight(self, value):
        self.__value = value
        return self

    def mutate(self):
        # minimum, maximum = self.__value_range
        # self.__value = (maximum - minimum) * np.random.sample() + minimum
        self.__value = np.random.choice(self.__value_space)

class GeneticAgent:
    def __init__(self, mutation_rate=0.10, weight_step=0.01):
        self.weight_step = weight_step
        self.mutation_rate = mutation_rate
        weights_count = len(ModelWeights.new._fields)
        
        gen_weight_set = lambda: [GeneticWeight(-1, 1, self.weight_step) for _ in range(7)]
        weights = tuple([gen_weight_set() for _ in range(weights_count)])
        self.__weights = ModelWeights.new(*weights)

    def __str__(self):
        repr_str = ""
        for i, field in enumerate(self.weights._fields):
            weight_set = [w.value for w in self.weights[i]]
            repr_str += f"weight.{field} = {weight_set}\n"
        
        return repr_str

    @property
    def weights_as_bundles(self):
        weight_bundles = []
        for weight_set in self.__weights:
            bundled_set = nb.typed.List.empty_list(GeneticWeightBundle.type)
            for weight in weight_set:
                bundled_set.append(weight.to_bundle())
            
            weight_bundles.append(bundled_set)
            
        return ModelWeights.new(*weight_bundles)
        # weight_bundles = [w.to_bundle() for w in self.__weights]
        # return ModelWeights.new(*weight_bundles)

    @property
    def weights(self):
        return self.__weights

    def set_weights(self, weights):
        new_weight = lambda value: GeneticWeight(-1, 1, self.weight_step).set_weight(value)

        new_weights = []
        if isinstance(weights[0][0], (int, float)):
            for weight_set in weights:
                new_weights.append([new_weight(float(w)) for w in weight_set])
        else:
            for weight_set in weights:
                new_weights.append([new_weight(float(w.value)) for w in weight_set])
    
        self.__weights = ModelWeights.new(*new_weights)
        return self

def load_trained_model(number=0):
    name = "weights.pyobj" if number == 0 else f"weights #{number}.pyobj"
    with open(f"resources//{name}", "rb+") as file:
        weights = pickle.load(file)
        return GeneticAgent().set_weights(weights)

if __name__ == "__main__":
    
    while True:
        try:
            depth = int(input("iter_depth: "))
            delay = int(input("delay: "))/1000
            m = int(input("model #: "))
            trained_model = load_trained_model(m)
            blocks = np.zeros((23,10), dtype=np.int8)
            py_simulate(blocks, trained_model, depth, delay)
        except KeyboardInterrupt:
            break
        finally:
            pass
