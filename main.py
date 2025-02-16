import random
import time

import numpy as np
import matplotlib.pyplot as plt
import math
from math import isclose

global distance_matrix


def parse_tsplib(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    nodes = []
    for line in lines:
        parts = line.strip().split()
        if parts[0].isdigit():
            nodes.append((float(parts[1]), float(parts[2])))

    distance_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            distance_matrix[i, j] = euclidean_distance(nodes[i], nodes[j])

    return nodes, distance_matrix


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def total_distance(tour):
    distance = 0
    for i in range(len(tour)):
        distance += distance_matrix[tour[i - 1], tour[i]]
    return distance

def initialize_population(pop_size):
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(pop_size)]

def roulette_wheel_selection(population, fitness):
    total_fitness = sum(fitness)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, f in enumerate(fitness):
        current += f
        if current >= pick:
            return population[i]


def tournament_selection(population, fitness, k=3):
    selected = random.sample(list(zip(population, fitness)), k)
    return min(selected, key=lambda x: x[1])[0]


def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    remaining = [gene for gene in parent2 if gene not in child]
    child = [remaining.pop(0) if gene == -1 else gene for gene in child]
    return child


def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = parent1[:]
    mapping = {parent1[i]: parent2[i] for i in range(start, end)}
    for i in range(start, end):
        child[i] = parent2[i]
    for i in range(size):
        if i < start or i >= end:
            while child[i] in mapping:
                child[i] = mapping[child[i]]
    return child


def swap_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour


def inversion_mutation(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i:j] = reversed(tour[i:j])
    return tour

def plot_tour(tour, nodes, best_dist, params=None):
    x = [nodes[i][0] for i in tour]
    y = [nodes[i][1] for i in tour]
    # Also add the first city to the end to complete the loop
    x.append(x[0])
    y.append(y[0])
    if params is None:
        plt.title(f"Best Distance: {best_dist}")
    else:
        plt.title(f"Best Distance: {best_dist}, \nParameters: {params}")
    plt.plot(x, y, 'o-')
    plt.show()

def genetic_algorithm(pop_size=200, generations=500, crossover_rate=0.9, mutation_rate=0.2):
    population = initialize_population(pop_size)
    best_fitness_over_time = []

    for generation in range(generations):
        genome_fitness = [total_distance(genome) for genome in population]
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, genome_fitness)
            parent2 = tournament_selection(population, genome_fitness)
            if random.random() < crossover_rate:
                child1 = ordered_crossover(parent1, parent2)
                child2 = ordered_crossover(parent2, parent1)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = inversion_mutation(child2)
            new_population.extend([child1, child2])
        population = new_population
        best_fitness = round(min(genome_fitness), 3)
        best_fitness_over_time.append(best_fitness)

        # Checking if fitness has stagnated
        if len(best_fitness_over_time) > 400:
            if isclose(best_fitness, best_fitness_over_time[-60], rel_tol=0.015):
                print(f'Stopping early at generation {generation} due to no improvement in fitness.')
                break

    best_loop_distance = min(genome_fitness)
    best_ga_tour = population[genome_fitness.index(best_loop_distance)]

    print(f"Best loop distance: {best_loop_distance}")
    plot_fitness_over_time(best_fitness_over_time)
    return best_ga_tour, best_loop_distance

def plot_fitness_over_time(fitness_scores):
    plt.plot(fitness_scores)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Time")
    plt.show()

# Gridsearch for the best possible hyperparameters
MUTATION_RATES = [0.01, 0.05, 0.12]
CROSSOVER_RATES = [0.7, 0.8, 0.9]
POPULATION_SIZES = [100, 200, 350]
# MUTATION_RATES = [0.01]
# CROSSOVER_RATES = [0.8]
# POPULATION_SIZES = [350]

def gridsearch(generations):
    best_params, best_solution = None, None
    best_ga_distance = float('inf')


    # Gridsearch for the best possible combination of hyperparameters
    # Hyperparameters: Population size, Mutation rate, Crossover rate
    for POPULATION_SIZE in POPULATION_SIZES:
        for MUTATION_RATE in MUTATION_RATES:
            for CROSSOVER_RATE in CROSSOVER_RATES:
                iter_start_time = time.time()
                params = {POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE}

                current_tour, current_distance = (
                    genetic_algorithm(pop_size=POPULATION_SIZE, generations=generations, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE))
                iter_time_sum = time.time() - iter_start_time
                print(
                    f'GA with Population size: {POPULATION_SIZE}, Mutation rate: {MUTATION_RATE}, Crossover rate: {CROSSOVER_RATE} \n Took: {iter_time_sum}')

                if current_distance < best_ga_distance:
                    plot_tour(current_tour, nodes, current_distance, params)
                    best_solution, best_ga_distance = current_tour, current_distance
                    best_params = (POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE)
                print('----------------------------------------------')

    print(f'Best Params - Pop:{best_params[0]}, Mut:{best_params[1]}, Crs: {best_params[2]}')
    print("Best Tour:", best_solution)
    print("Best Distance:", best_ga_distance)
    return

NUM_CITIES = 100

if __name__ == "__main__":
    # filename = "berlin52.txt"
    filename = "kroA100.txt"

    nodes, distance_matrix = parse_tsplib(filename)
    start_time = time.time()

    # gridsearch(generations=410)

    best_tour, best_distance = genetic_algorithm(generations=900)
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)

    end_time = time.time()
    plot_tour(best_tour, nodes, best_distance)
    print("Computation time taken:", end_time - start_time, "seconds")

    print("\n--------------\n")

    start_time = time.time()
    best_tour, best_distance = genetic_algorithm(generations=350)
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)

    end_time = time.time()
    plot_tour(best_tour, nodes, best_distance)
    print("Computation time taken:", end_time - start_time, "seconds")
