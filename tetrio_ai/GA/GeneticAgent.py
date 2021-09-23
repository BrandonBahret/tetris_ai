import numpy as np
import numba as nb

from copy import deepcopy

from GA.GeneticWeight import GeneticWeight
from Types.NumbaDefinitions import ModelWeights, GeneticWeightBundle

def bundle_population(population):
    bundles = nb.typed.List.empty_list(ModelWeights.type)
    for agent in population:
        bundle = agent.weights_as_bundles
        bundles.append(bundle)
    
    return bundles

class GeneticAgent:
    def __init__(self, mutation_rate=0.10, weight_step=0.01):
        self.weight_step = weight_step
        self.mutation_rate = mutation_rate
        weights_count = len(ModelWeights.new._fields)
        
        gen_weight_set = lambda: [GeneticWeight(-1, 10, self.weight_step) for _ in range(5)]
        weights = tuple([gen_weight_set() for _ in range(weights_count)])
        self.__weights = ModelWeights.new(*weights)

    def __str__(self):
        repr_str = ""
        for i, field in enumerate(self.weights._fields):
            weight_set = [w.value for w in self.weights[i]]
            repr_str += f"weight.{field} = {weight_set}\n"
        
        return repr_str

    def copy(self):
        return deepcopy(self)

    def create_child(self, other):
        child = deepcopy(self)

        should_mutate = lambda: np.random.sample() > (1 - self.mutation_rate)
        should_copy = lambda: np.random.sample() > 0.5

        for set_i, weight_set in enumerate(other.weights):
            for weight_i, weight in enumerate(weight_set):
                if should_copy():
                    child.weights[set_i][weight_i].set_weight(weight.value)

                if should_mutate():
                    child.weights[set_i][weight_i].mutate()

        return child

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
        new_weight = lambda value: GeneticWeight(-1, 10, self.weight_step).set_weight(value)

        new_weights = []
        if isinstance(weights[0][0], (int, float)):
            for weight_set in weights:
                new_weights.append([new_weight(float(w)) for w in weight_set])
        else:
            for weight_set in weights:
                new_weights.append([new_weight(float(w.value)) for w in weight_set])
    
        self.__weights = ModelWeights.new(*new_weights)
        return self
