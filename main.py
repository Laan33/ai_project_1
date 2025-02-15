import random
import numpy as np
import matplotlib.pyplot as plt
import math


def parse_tsplib(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    nodes = []
    dimension = 0
    for line in lines:
        parts = line.strip().split()
        if parts[0] == "DIMENSION:":
            dimension = int(parts[1])
        elif parts[0].isdigit():
            nodes.append((float(parts[1]), float(parts[2])))
    return nodes


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def total_distance(tour, nodes):
    return sum(
        euclidean_distance(nodes[tour[i]], nodes[tour[i + 1]]) for i in range(len(tour) - 1)) + euclidean_distance(
        nodes[tour[-1]], nodes[tour[0]])


def initialize_population(pop_size, num_cities):
    return [random.sample(range(num_cities), num_cities) for _ in range(pop_size)]


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


def genetic_algorithm(filename, pop_size=100, generations=500, crossover_rate=0.8, mutation_rate=0.2):
    nodes = parse_tsplib(filename)
    num_cities = len(nodes)
    population = initialize_population(pop_size, num_cities)
    best_fitness_over_time = []

    for _ in range(generations):
        fitness = [1 / total_distance(tour, nodes) for tour in population]
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)
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
        best_fitness_over_time.append(max(fitness))

    best_tour = min(population, key=lambda tour: total_distance(tour, nodes))
    best_distance = total_distance(best_tour, nodes)
    plot_fitness_over_time(best_fitness_over_time)
    return best_tour, best_distance

def plot_fitness_over_time(fitness_scores):
    plt.plot(fitness_scores)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Time")
    plt.show()

if __name__ == "__main__":
    filename = "berlin52.txt"
    best_tour, best_distance = genetic_algorithm(filename)
    print("Best Tour:", best_tour)
    print("Best Distance:", best_distance)
