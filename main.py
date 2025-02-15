import random
import time

import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Constants
POPULATION_SIZE = 400 # Number of individuals in the population
GENOME_SIZE = 52  # Number of cities
MUTATION_RATE = 0.12 # Chance of mutation per gene
CROSSOVER_RATE = 0.88 # Chance of crossover for a pair of parents
GENERATIONS = 2500 # Number of generations to run the algorithm


def calculate_best_path(coordinates):
    all_permutations = permutations(range(GENOME_SIZE))
    best_path = None
    best_distance = float('inf')
    perm_count = 0
    for perm in all_permutations:
        perm_count += 1
        current_distance = 0
        for i in range(GENOME_SIZE):
            current_distance += np.sqrt(
                (coordinates[perm[i - 1]][0] - coordinates[perm[i]][0]) ** 2 +
                (coordinates[perm[i - 1]][1] - coordinates[perm[i]][1]) ** 2
            )
        if current_distance < best_distance:
            best_distance = round(current_distance, 3)
            best_path = perm
        if perm_count % 1000 == 0:
            print(f'Current at perm count: {perm_count} permutation: {perm}, Distance: {current_distance}')

    return best_path, best_distance

def load_distance_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header lines until NODE_COORD_SECTION
    start_index = lines.index('NODE_COORD_SECTION\n') + 1
    coordinates = []

    for line in lines[start_index:]:
        if line.strip() == 'EOF':
            break
        parts = line.strip().split()
        coordinates.append((float(parts[1]), float(parts[2])))

    # Calculate distance matrix
    num_cities = len(coordinates)
    matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist = np.sqrt(
                    (coordinates[i][0] - coordinates[j][0]) ** 2 + (coordinates[i][1] - coordinates[j][1]) ** 2)
                matrix[i][j] = dist

    return matrix, coordinates

def load_coordinates(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Skip header lines until NODE_COORD_SECTION
    start_index = lines.index('NODE_COORD_SECTION\n') + 1
    coordinates = []

    for line in lines[start_index:]:
        if line.strip() == 'EOF':
            break
        parts = line.strip().split()
        coordinates.append((float(parts[1]), float(parts[2])))

    print("Coordinates", coordinates)
    plot_coordinates(coordinates)
    # num_cities = len(coordinates)
    return coordinates

def plot_coordinates(coordinates):
    x = [c[0] for c in coordinates]
    y = [c[1] for c in coordinates]
    plt.plot(x, y, 'o')
    plt.show()

def plot_tour(coordinates, tour, best_ga_fitness, best_possible_distance):
    x = [coordinates[i][0] for i in tour]
    y = [coordinates[i][1] for i in tour]
    # Also add the first city to the end to complete the tour
    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, 'o-')
    plt.title(f'Best GA Fitness: {best_ga_fitness}, Best Possible Distance: {best_possible_distance}')
    plt.show()

# Initialize population with random permutations of cities
def init_population(population_size, genome_size):
    population = []
    for _ in range(population_size):
        genome = list(range(genome_size))
        random.shuffle(genome)
        population.append(genome)
    return population

# Fitness function: total distance of the tour
def fitness(genome, distance_matrix):
    total_distance = 0
    for i in range(len(genome)):
        total_distance += distance_matrix[genome[i - 1]][genome[i]]
    return -total_distance  # Negative because we want to minimise distance

# Selection mechanism: roulette wheel selection
def select_parent(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for individual, individual_fitness in zip(population, fitnesses):
        cumulative_fitness += individual_fitness
        if cumulative_fitness <= pick: # Smaller or equal to pick because values are negative
            return individual
    print(f"Total fitness: {total_fitness}, Pick: {pick}, Cumulative fitness: {cumulative_fitness}")
    print("Parent not found")
    return population[-1]

def tournament_selection(population, fitnesses, tournament_size=5):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    selected.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (higher is better)
    return selected[0][0]  # Return best individual

# Order Crossover (OX)
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child1 = [None] * len(parent1)
        child1[start:end] = parent1[start:end]
        child2 = [None] * len(parent2)
        child2[start:end] = parent2[start:end]

        fill_child(child1, parent2, end)
        fill_child(child2, parent1, end)

        return child1, child2
    else:
        return parent1, parent2



def fill_child(child, parent, end):
    current_pos = end
    for gene in parent:
        if gene not in child:
            if current_pos >= len(child):
                current_pos = 0
            child[current_pos] = gene
            current_pos += 1

# Swap mutation
def swap_mutate(genome):
    for i in range(len(genome)):
        if random.random() < MUTATION_RATE:
            j = random.randint(0, len(genome) - 1)
            genome[i], genome[j] = genome[j], genome[i]
    return genome

# Displacement mutation
def displacement_mutate(genome):
    if random.random() < MUTATION_RATE:
        start, end = sorted(random.sample(range(len(genome)), 2))
        displaced = genome[start:end]
        genome[start:end] = []
        insert_index = random.randint(0, len(genome))
        genome[insert_index:insert_index] = displaced
    return genome

# Genetic algorithm
def genetic_algorithm(distance_matrix, coordinates, best_possible_distance = None):
    global fitness_values
    population = init_population(POPULATION_SIZE, GENOME_SIZE)

    for generation in range(GENERATIONS):
        fitness_values = [fitness(genome, distance_matrix) for genome in population]

        new_population = []
        for _ in range(POPULATION_SIZE // 2): # Half of the (not so good) population scrapped
            # parent1 = select_parent(population, fitness_values)
            # parent2 = select_parent(population, fitness_values)
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            offspring1, offspring2 = crossover(parent1, parent2)

            new_population.extend([swap_mutate(offspring1), swap_mutate(offspring2)])

        population = new_population
        fitness_values = [fitness(genome, distance_matrix) for genome in population]

    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]
    best_fitness = round(-fitness(best_solution, distance_matrix))
    plot_fitness_over_generations(fitness_values)
    plot_tour(coordinates, best_solution, best_fitness, best_possible_distance)
    print(f'Best GA Solution: {best_solution}')
    print(f'Best GA Fitness: {best_fitness}')

# Plotting fitness over generations
def plot_fitness_over_generations(fitness_val):
    plt.plot(fitness_val)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over generations')
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    distance_matrix, coordinates = load_distance_matrix('berlin52.txt')
    # distance_matrix, coordinates = load_distance_matrix('kroA100.txt')
    # distance_matrix, coordinates = load_distance_matrix('pr1002.txt')
    # best_possible_path, best_possible_distance = calculate_best_path(coordinates)
    # print("Best possible path distance:", best_possible_distance)
    # print("Best possible path:", best_possible_path)


    genetic_algorithm(distance_matrix, coordinates)
    end_time = time.time()
    print("Computation time taken:", end_time - start_time, "seconds")





