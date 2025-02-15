import random
import time
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# Constants
GENOME_SIZE = 52  # Number of cities
GENERATIONS = 600 # Number of generations to run the algorithm

# POPULATION_SIZE = 400 # Number of individuals in the population
# MUTATION_RATE = 0.05 # Chance of mutation per gene
# CROSSOVER_RATE = 0.88 # Chance of crossover for a pair of parents

# Gridsearch for the best possible hyperparameters
# MUTATION_RATES = [0.01, 0.05, 0.12]
# CROSSOVER_RATES = [0.7, 0.8, 0.9]
# POPULATION_SIZES = [100, 200, 350]
MUTATION_RATES = [0.01,]
CROSSOVER_RATES = [0.7]
POPULATION_SIZES = [350]


def calculate_best_path(coordinates):
    all_permutations = permutations(range(GENOME_SIZE))
    best_path = None
    best_possible_distance = float('inf')
    perm_count = 0
    for perm in all_permutations:
        perm_count += 1
        current_distance = 0
        for i in range(GENOME_SIZE):
            current_distance += np.sqrt(
                (coordinates[perm[i - 1]][0] - coordinates[perm[i]][0]) ** 2 +
                (coordinates[perm[i - 1]][1] - coordinates[perm[i]][1]) ** 2
            )
        if current_distance < best_possible_distance:
            best_possible_distance = round(current_distance, 3)
            best_path = perm
        if perm_count % 1000 == 0:
            print(f'Current at perm count: {perm_count} permutation: {perm}, Distance: {current_distance}')

    return best_path, best_possible_distance

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

def plot_tour(coordinates, tour, best_ga_fitness):
    x = [coordinates[i][0] for i in tour]
    y = [coordinates[i][1] for i in tour]
    # Also add the first city to the end to complete the tour
    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, 'o-')
    plt.title(f'Best GA Fitness: {best_ga_fitness}')
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
    return total_distance  # Positive because we want to minimise distance

# Selection mechanism: roulette wheel selection
def roulette_select_parent(population, fitnesses):
    min_fitness = min(fitnesses)
    adjusted_fitnesses = [f - min_fitness for f in fitnesses]  # Adjust fitness values
    total_fitness = sum(adjusted_fitnesses)
    pick = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for individual, adjusted_fitness in zip(population, adjusted_fitnesses):
        cumulative_fitness += adjusted_fitness
        if cumulative_fitness >= pick:
            return individual
    return population[-1]

def tournament_selection(population, fitnesses, tournament_size=5):
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)
    # Sort by fitness (lower is better)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]  # Return best individual

# Order Crossover (OX)
def order_crossover(parent1, parent2):
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

def pmx_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)

    # Copy the crossover segment from parent1
    child[start:end] = parent1[start:end]

    mapping = {parent1[i]: parent2[i] for i in range(start, end)}

    # Fill the remaining positions
    for i in range(len(parent1)):
        if child[i] is None:
            val = parent2[i]
            while val in mapping:
                val = mapping[val]
            child[i] = val

    return child


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
def genetic_algorithm():
    for generation in range(GENERATIONS):
        population = init_population(POPULATION_SIZE, GENOME_SIZE)
        fitness_values = [fitness(genome, distance_matrix) for genome in population]

        # **Elitism** - Keep the top 5% of solutions
        elite_size = int(POPULATION_SIZE * 0.05)
        elite_indices = np.argsort(fitness_values)[:elite_size]  # Get top N indices
        elite = [population[i] for i in elite_indices]

        new_population = elite  # Keep the elites

        # Generate the rest of the population
        for _ in range((POPULATION_SIZE - elite_size) // 2):
            # parent1 = tournament_selection(population, fitness_values)
            # parent2 = tournament_selection(population, fitness_values)
            parent1 = roulette_select_parent(population, fitness_values)
            parent2 = roulette_select_parent(population, fitness_values)
            offspring1 = pmx_crossover(parent1, parent2)
            offspring2 = pmx_crossover(parent2, parent1)

            new_population.extend([swap_mutate(offspring1), swap_mutate(offspring2)])

        population = new_population
        fitness_values = [fitness(genome, distance_matrix) for genome in population]

    best_index = fitness_values.index(min(fitness_values))
    best_solution_loop = population[best_index]
    best_fitness = round(fitness(best_solution_loop, distance_matrix))
    plot_fitness_over_generations(fitness_values)
    print(f'Best GA Solution: {best_solution_loop}')
    print(f'Best GA Fitness: {best_fitness}')
    return best_solution_loop, best_fitness

# Plotting fitness over generations
def plot_fitness_over_generations(fitness_val):
    plt.plot(fitness_val)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over generations\n, Pop size: {}, Mut rate: {}, Cross rate: {}'.format(POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE))
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    distance_matrix, coordinates = load_distance_matrix('berlin52.txt')
    # distance_matrix, coordinates = load_distance_matrix('kroA100.txt')
    # distance_matrix, coordinates = load_distance_matrix('pr1002.txt')

    best_solution, best_distance = float('inf'), float('inf')
    best_params = None

    # Gridsearch for the best possible combination of hyperparameters
    # Hyperparameters: Population size, Mutation rate, Crossover rate
    for POPULATION_SIZE in POPULATION_SIZES:
        for MUTATION_RATE in MUTATION_RATES:
            for CROSSOVER_RATE in CROSSOVER_RATES:
                print(f'Running GA with Population size: {POPULATION_SIZE}, Mutation rate: {MUTATION_RATE}, Crossover rate: {CROSSOVER_RATE}')
                current_solution, current_distance = genetic_algorithm()
                if current_distance < best_distance:
                    plot_tour(coordinates, current_solution, current_distance)
                    best_solution, best_distance = current_solution, current_distance
                    best_params = (POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE)
                print('----------------------------------------------')


    # genetic_algorithm(distance_matrix, coordinates)
    end_time = time.time()
    print("Computation time taken:", end_time - start_time, "seconds")
    print("Best GA Solution distance: ", best_distance)
    print("Best solution params: ", best_params)





