import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from functools import partial
import pickle
import copy

from threading import Thread

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc 

import numba as nb
import numpy as np
import random

from GA.GeneticAgent import GeneticAgent, bundle_population
from GA.Simulation import simulate

from typing import NamedTuple
from Types.NumbaDefinitions import NbExpose
from Types.NumbaDefinitions import i1MatrixContType
from Types.NumbaDefinitions import ModelWeights, GeneticWeightBundle
import math

from frame_display import FrameDisplay
from FrameGrabber.Frame import Frame
from tqdm import tqdm
# from py_simulate import py_simulate
# import pygame

np_unichar = np.dtype('<U1')
UnicharType = nb.from_dtype(np_unichar)
UnicharListType = nb.types.ListType(UnicharType)
unicode_lsttype = nb.types.ListType(nb.types.unicode_type)

def save_agent(agent):
    base_dir = os.path.dirname(os.getcwd())
    with open(f"{base_dir}\\resources\\weights.pyobj", "wb+") as file:
        weight_sets = []
        for weight_set in agent.weights:
            weights = [w.value for w in weight_set]
            weight_sets.append(weights)

        pickle.dump(weight_sets, file)

def load_trained_model(number=0):
    base_dir = os.path.dirname(os.getcwd())
    name = "weights.pyobj" if number == 0 else f"weights #{number}.pyobj"
    with open(f"{base_dir}\\resources\\{name}", "rb+") as file:
        weights = pickle.load(file)
        return GeneticAgent().set_weights(weights)

def load_inputs():
    base_dir = os.path.dirname(os.getcwd())
    with open(f"{base_dir}\\resources\\simulation_inputs.pyobj", "rb+") as file:
        sim_inputs = nb.typed.List(pickle.load(file))
    
    return sim_inputs

@NbExpose
class WeightDistance(NamedTuple):
    distance: nb.float64
    value: nb.float64

@nb.njit(WeightDistance.type(GeneticWeightBundle.lsttype))
def calculate_max_distance(weight_set):
    maximum = weight_set[0].maximum
    minimum = weight_set[0].minimum
    weight_set_mean = np.mean(np.asarray([w.value for w in weight_set]))

    d1 = abs(minimum - weight_set_mean)
    d2 = abs(maximum - weight_set_mean)

    distance = max(d1, d2)
    value = minimum if d1 > d2 else maximum

    return WeightDistance.new(distance, value)

@nb.njit(nb.float64(ModelWeights.lsttype, ModelWeights.type))
def measure_diversity(population, agent_weights):
    weights_length = len(agent_weights)
    diversity_scores = np.zeros((weights_length,), dtype=np.float64)
    weight_set_mean = lambda weight_set: np.mean(np.asarray([w.value for w in weight_set]))

    for i in range(weights_length):
        for each_weights in population:
            weight_distance = abs(weight_set_mean(agent_weights[i]) - weight_set_mean(each_weights[i]))
            max_distance = calculate_max_distance(each_weights[i]).distance
            diversity_scores[i] += weight_distance / max_distance

    for i in range(len(diversity_scores)):
        diversity_scores[i] /= len(population)

    return np.mean(diversity_scores)

@nb.njit(ModelWeights.lsttype(ModelWeights.lsttype, nb.int64))
def get_sample_population(population, sample_size):
    pop_size = len(population)
    sample_size = min(pop_size, sample_size)
    sample_indices = np.random.choice(np.arange(pop_size), sample_size, replace=False)

    sample = nb.typed.List.empty_list(ModelWeights.type)
    for i in sample_indices:
        sample.append(population[i])

    return sample

@nb.njit
def stochastic_measure_diversity(population, agent, sample_rate=0.5):
    pop_size = len(population)
    sample_size = int(pop_size * sample_rate)
    pop_sample = get_sample_population(population, sample_size)
    return measure_diversity(pop_sample, agent)

@nb.njit
def measure_fitness(blocks, sim_inputs, population, agent, use_diversity_rank=True, iteration_depth=75):
    metrics = simulate(blocks, int(iteration_depth), agent, sim_inputs)

    fitness_score = 0
    if metrics.gameover:
        fitness_score -= 256

    fitness_score += 5 * metrics.line_count
    fitness_score += 10 * metrics.tetris_count
    fitness_score -= metrics.final_board_height
    fitness_score -= metrics.created_hole_count

    max_score = (metrics.max_iters // 5) * 5

    if use_diversity_rank:
        diversity_score = measure_diversity(population, agent)
        return (diversity_score * fitness_score) / max_score
    else:
        return fitness_score / max_score

@NbExpose
class ScoredWeights(NamedTuple):
    weights: ModelWeights.type
    fitness: nb.float64

@nb.njit(ScoredWeights.lsttype(i1MatrixContType, unicode_lsttype, ModelWeights.lsttype, nb.boolean, nb.int64), parallel=True)
def impl_score_population(blocks, sim_inputs, population, use_diversity_rank, iteration_depth):
    pop_size = len(population)

    scored_weights = nb.typed.List.empty_list(ScoredWeights.type)
    for i in nb.prange(pop_size):
        agent = population[i]
        fitness_score = measure_fitness(blocks, sim_inputs, population, agent, use_diversity_rank, iteration_depth)
        scored_weight = ScoredWeights.new(agent, fitness_score)
        scored_weights.append(scored_weight)

    return scored_weights

def score_population(blocks, sim_inputs, population, use_diversity_rank=True, iteration_depth=75):
    if population is None:
        return None
    
    population_bundle = bundle_population(population)
    scored_weights = impl_score_population(blocks, sim_inputs, population_bundle, use_diversity_rank, iteration_depth)
    return sorted(scored_weights, key= lambda pair: pair.fitness, reverse=True)

def reproduce(population, target_pop_size):
    new_population = []
    pop_size = len(population)

    if pop_size < 2:
        return None

    while len(new_population) < target_pop_size:
        p1 = population[0]

        for p2 in population[1:]:
            if len(new_population) >= target_pop_size:
                break
            
            child = p1.create_child(p2)
            new_population.append(child)

    return new_population

def reproduce_pairwise(population, target_pop_size):    
    pop_size = len(population)
    new_population = []

    target_pop_size = max(2, target_pop_size)
    if pop_size < 2:
        print(f"{pop_size=}")
        return None

    i = 0
    while len(new_population) < target_pop_size:
        p1 = population[i]
        p2 = population[i+1]
        child = p1.create_child(p2)

        new_population.append(child)
        i = (i + 2) % (pop_size-1)

    return new_population

def reproduce_randomized(population, target_pop_size):
    new_population = []

    while len(new_population) < target_pop_size:
        p1, p2 = np.random.choice(population, 2, replace=False)
        child = p1.create_child(p2)
        new_population.append(child)

    return new_population

def tournament_selection(scored_pop, thresh=0.5, bracket_size=4):
    pop_size = len(scored_pop)
    indices = np.arange(pop_size, dtype=np.int64)
    target_size = max(2, int(pop_size * thresh))
    bracket_size = max(2, bracket_size)
    bracket_size = min(pop_size, bracket_size)

    selected_agents = []
    while len(selected_agents) < target_size:
        bracket_is = np.random.choice(indices, bracket_size, replace=False)
        bracket = [scored_pop[i] for i in bracket_is]
        bracket = sorted(bracket, key= lambda pair: pair.fitness, reverse=True)
        best_agent = bracket[0]
        selected_agents.append(best_agent)

    return sorted(selected_agents, key= lambda pair: pair.fitness, reverse=True)

def threshold_selection(scored_pop, thresh=0.5):
    pop_size = len(scored_pop)
    target_size = max(2, int(pop_size * thresh))
    selected_agents = scored_pop[:target_size]
    return selected_agents

def roulette_selection(scored_pop, thresh=0.5):
    pop_size = len(scored_pop)
    target_size = max(2, int(pop_size * thresh))

    fitness_scores = [pair.fitness for pair in scored_pop]
    fitness_sum = sum(fitness_scores)
    fitness_sum = fitness_sum if fitness_sum != 0 else 1

    calculate_probablity = lambda fitness_score: max(0.001, fitness_score / fitness_sum)
    probabilities = [calculate_probablity(f) for f in fitness_scores]

    selected_agents = []
    i = -1
    while len(selected_agents) < target_size:
        i = (i+1) % pop_size

        if np.random.sample() > (1 - probabilities[i]):
            agent = scored_pop[i]
            selected_agents.append(agent)

    return sorted(selected_agents, key= lambda pair: pair.fitness, reverse=True)


def unbundle_agent(scored_agent, mutation_rate=0.01, weight_step=0.01):
    unbundled_weights = []
    for weight_set in scored_agent.weights:
        weights = [float(w.value) for w in weight_set]
        unbundled_weights.append(weights)

    agent = GeneticAgent(mutation_rate, weight_step).set_weights(unbundled_weights)
    return agent

def unbundle_population(scored_pop, mutation_rate=0.01, weight_step=0.01):
    mr, ws = mutation_rate, weight_step
    unbundle = partial(unbundle_agent, mutation_rate=mr, weight_step=ws)
    return [unbundle(scored_agent) for scored_agent in scored_pop]

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

class GeneticModelTrainer:
    def __init__(self, blocks_set, base_pop_size, mutation_rate, weight_step, iteration_depth=None):
        self.mutation_rate = float(mutation_rate)
        self.weight_step = float(weight_step)
        self.base_pop_size = int(base_pop_size)
        self.iteration_depth = iteration_depth
        self.inputs = load_inputs()

        print(f"{mutation_rate=:.4f}, {weight_step=:.4f}, {base_pop_size=}")

        self.blocks_set = blocks_set

        self.overall_scores = []
        self.each_generation_scores = []
        self.best_generation_scores = []
        self.best_score = -np.inf

        self.trained_agent = None
        self.disp = None

    def gen_base_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            agent = GeneticAgent(self.mutation_rate, self.weight_step)
            population.append(agent)
        
        return population

    def create_new_generation(self, scored_pop, mutation_rate=None, weight_step=None):
        select_non_none = lambda a, b: a if a is not None else b
        mutation_rate = select_non_none(mutation_rate, self.mutation_rate)
        weight_step = select_non_none(weight_step, self.weight_step)

        pop_size = self.base_pop_size
        pop_sample = threshold_selection(scored_pop, thresh=0.55)
        pop_sample += roulette_selection(scored_pop, thresh=0.05)
        
        pop_sample = unbundle_population(pop_sample, mutation_rate, weight_step)
        new_generation = reproduce(pop_sample, pop_size)
        new_generation += reproduce_randomized(pop_sample, pop_size//4)
        return new_generation

    def plot_training_progress(self):
        xs = list(range(len(self.overall_scores)))
        fig = plt.figure(dpi=200)
        plt.plot(xs[1:], self.best_generation_scores[1:], 'red')
        plt.plot(xs[1:], self.each_generation_scores[1:], 'blue')
        plt.plot(xs[1:], self.overall_scores[1:], 'orange')

        plt.title(f"score: {self.best_score:0.2f}")
        plt.xlabel("GENERATION")
        plt.ylabel("FITNESS")

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_img = Frame.resize(data, 400)

        fig.clf()
        plt.close()
        gc.collect()

        if self.disp is None:
            self.disp = FrameDisplay(plot_img)
        else:
            self.disp.update(plot_img)

    def train(self, generations):
        compare_fitness = lambda agent: agent.fitness
        base_pop = self.gen_base_population(self.base_pop_size)
        scored_pop = score_population(self.blocks_set[0], self.inputs, base_pop, True)
        top_agent_hist = scored_pop[:2]
        top_agent_hist_pop = base_pop[:2]
        self.plot_training_progress()

        scaled_depth = lambda g: int(min(100, (g*0.25)+20))
        constant_depth = lambda g: int(self.iteration_depth)
        iter_depth = scaled_depth if self.iteration_depth is None else constant_depth

        try:
            for g in tqdm(range(generations)):
                if self.disp is not None:
                    if not self.disp.running:
                        break
                    # for e in self.disp.get_events():
                    #     if e.type == pygame.MOUSEBUTTONDOWN:
                    #         # Thread(target=py_simulate, args=(self.blocks_set[0], self.trained_agent.weights, 100, 100,)).start()
                    #         py_simulate(self.blocks_set[0], self.trained_agent.weights, 100, 100)

                blocks = random.choice(self.blocks_set)
                uses_diversity = g <= (generations * 0.75)

                scored_pop = score_population(blocks, self.inputs, base_pop, uses_diversity, iteration_depth=iter_depth(g))
                scored_top_hist = score_population(blocks, self.inputs, top_agent_hist_pop, uses_diversity, iteration_depth=iter_depth(g))

                best_agent = scored_pop[0]
                hist_top_agent = scored_top_hist[0]
                generations_best_agent = max(hist_top_agent, best_agent, key=compare_fitness)

                base_pop = self.create_new_generation(scored_pop)
                top_agent_hist_pop = self.create_new_generation(top_agent_hist)

                top_agent_hist.append(generations_best_agent)
                top_agent_hist = sorted(top_agent_hist, key= lambda pair: pair.fitness, reverse=True)
                top_agent_hist = top_agent_hist[:self.base_pop_size]
                if generations_best_agent.fitness > self.best_score:
                    self.best_score = generations_best_agent.fitness
                    self.trained_agent = copy.copy(generations_best_agent)

                self.overall_scores.append(self.best_score)
                self.each_generation_scores.append(best_agent.fitness)
                self.best_generation_scores.append(hist_top_agent.fitness)
                self.plot_training_progress()

                if square_wave(g, generations*.10, 1):
                    if self.trained_agent is not None:
                        save_agent(self.trained_agent)

        except KeyboardInterrupt:
            self.plot_training_progress()
        except BaseException as e:
            print(e)

        save_agent(self.trained_agent)
        return self.trained_agent
    