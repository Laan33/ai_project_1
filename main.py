import random
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from math import isclose

global distance_matrix

# Save results to a CSV file with headers for population size, mutation rate, crossover rate, etc.
def save_to_csv(csv_name, data):
    with open(csv_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Population Size", "Mutation Rate", "Crossover Rate", "Best Fitness", "Time", "Iterations"])
        writer.writerows(data)
    print(f"Results saved to {csv_name}")

# Parse TSPLIB format files and create distance matrix
def parse_tsplib(dataset_filename):
    with open("datasets/" + dataset_filename + ".txt", 'r') as file:
        lines = file.readlines()

    # Extract coordinates for each city
    nodes = []
    for line in lines:
        parts = line.strip().split()
        if parts[0].isdigit():
            nodes.append((float(parts[1]), float(parts[2])))

    # Calculate distances between all pairs of cities
    distance_matrix = np.zeros((len(nodes), len(nodes)))
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            distance_matrix[i, j] = euclidean_distance(nodes[i], nodes[j])

    return nodes, distance_matrix

# Calculate Euclidean distance between two points
def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# Calculate total distance of a tour
def total_distance(tour):
    distance = 0
    for i in range(len(tour)):
        distance += distance_matrix[tour[i - 1], tour[i]]
    return distance

# Create initial random population
def initialize_population(pop_size):
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(pop_size)]

# Tournament selection: select the best individual from k random candidates
def tournament_selection(population, fitness, k=3):
    selected = random.sample(list(zip(population, fitness)), k)
    return min(selected, key=lambda x: x[1])[0]

# Monte Carlo selection based on inverse fitness (not currently used)
def monte_carlo_selection(population, fitness):
    fitness = np.array(fitness)
    probabilities = 1 / (fitness + 1e-10)
    probabilities /= probabilities.sum()
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]

# Order Crossover (OX) operator
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    # Copy a segment from parent1
    child[start:end] = parent1[start:end]
    # Fill remaining positions with genes from parent2
    remaining = [gene for gene in parent2 if gene not in child]
    child = [remaining.pop(0) if gene == -1 else gene for gene in child]
    return child

# Partially Mapped Crossover (PMX) operator
def partially_mapped_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start:end] = parent1[start:end]
    # Create mapping between segments
    mapping = {}
    for i in range(start, end):
        mapping[parent1[i]] = parent2[i]
    # Fill remaining positions
    for i in range(size):
        if child[i] == -1:
            value = parent2[i]
            while value in child:
                value = mapping[value]
            child[i] = value
    return child

# Swap Mutation: swap two random cities
def swap_mutation(tour):
    i, j = random.sample(range(len(tour)), 2)
    tour[i], tour[j] = tour[j], tour[i]
    return tour

# Inversion Mutation: reverse a random segment
def inversion_mutation(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i:j] = reversed(tour[i:j])
    return tour

# Plot the tour on a 2D graph
def plot_tour(tour, nodes, best_dist, params=None):
    x = [nodes[i][0] for i in tour]
    y = [nodes[i][1] for i in tour]
    # add the first city to the end to complete the loop
    x.append(x[0])
    y.append(y[0])
    if params is None:
        plt.title(f"Best Distance: {best_dist}")
    else:
        plt.title(f"Best Distance: {best_dist}, \nParameters: {params}")
    plt.plot(x, y, 'o-')
    plt.show()

# Main genetic algorithm implementation
def genetic_algorithm(pop_size=185, generations=500, crossover_rate=0.87, mutation_rate=0.15, mut_mtd=None, crs_mtd=None):
    if crs_mtd is None:
        crs_mtd = partially_mapped_crossover
    if mut_mtd is None:
        mut_mtd = inversion_mutation

    population = initialize_population(pop_size)
    best_fitness_over_time = []

    # Main evolution loop
    for generation in range(generations):
        genome_fitness = [total_distance(genome) for genome in population]
        new_population = []

        # Create new population through selection, crossover, and mutation
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, genome_fitness)
            parent2 = tournament_selection(population, genome_fitness)

            # Apply crossover
            if random.random() < crossover_rate:
                child1 = crs_mtd(parent1, parent2)
                child2 = crs_mtd(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Apply mutation
            if random.random() < mutation_rate:
                child1 = mut_mtd(child1)
            if random.random() < mutation_rate:
                child2 = mut_mtd(child2)

            new_population.extend([child1, child2])

        population = new_population
        best_fitness = round(min(genome_fitness), 1)
        best_fitness_over_time.append(best_fitness)

        # Early stopping if no improvement
        if len(best_fitness_over_time) > 400:
            if isclose(best_fitness, best_fitness_over_time[-60], rel_tol=0.0025):
                print(f'Stopping early at generation {generation} due to no improvement in fitness.')
                break

    best_loop_distance = min(genome_fitness)
    best_ga_tour = population[genome_fitness.index(best_loop_distance)]
    print(f"Best loop distance: {round(best_loop_distance, 1)}")
    plot_fitness_over_time(best_fitness_over_time)
    return best_ga_tour, best_loop_distance, generation

# Plot fitness progress over generations
def plot_fitness_over_time(fitness_scores):
    plt.plot(fitness_scores)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Time")
    plt.show()

# Grid search parameters
MUTATION_RATES = [0.03, 0.12, 0.2]
CROSSOVER_RATES = [0.6, 0.71, 0.82, 0.9]
POPULATION_SIZES = [120, 280, 470]

# Perform grid search over parameters
def gridsearch(generations, filename):
    grid_results = []
    for POPULATION_SIZE in POPULATION_SIZES:
        for MUTATION_RATE in MUTATION_RATES:
            for CROSSOVER_RATE in CROSSOVER_RATES:
                start_time = time.time()
                current_tour, current_distance, generation = genetic_algorithm(
                    pop_size=POPULATION_SIZE,
                    generations=generations,
                    crossover_rate=CROSSOVER_RATE,
                    mutation_rate=MUTATION_RATE
                )
                elapsed_time = round((time.time() - start_time), 3)
                grid_results.append((POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE,
                                  current_distance, elapsed_time, generation, current_tour))
    save_to_csv("results/" + filename + "_results.csv", grid_results)
    return grid_results

# Display the best result from grid search
def show_best_result():
    best_result = min(results, key=lambda x: x[3])
    best_population_size, best_mutation_rate, best_crossover_rate, best_distance, best_time, best_generations, best_tour = best_result
    print(f"Best Result:\nPopulation Size: {best_population_size}\n"
          f"Mutation Rate: {best_mutation_rate}\n"
          f"Crossover Rate: {best_crossover_rate}\n"
          f"Best Distance: {round(best_distance, 1)}\n"
          f"Time: {best_time}\nGenerations: {best_generations}")
    plot_tour(best_tour, nodes, best_distance)

# Load results from a CSV file
def load_best_result(results_file):
    with open("results/" + results_file + "_results.csv", "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        results = [tuple(map(float, row)) for row in reader]
    return results

# Number of cities in the problem
NUM_CITIES = 1002

if __name__ == "__main__":
    # Select dataset
    filename = "berlin52"
    # filename = "kroA100"
    # filename = "pr1002"
    nodes, distance_matrix = parse_tsplib(filename)

    # Run grid search
    results = gridsearch(generations=400, filename=filename)
    show_best_result()

    # Alternative options (commented out):
    # Load results from CSV
    # results = load_best_result(filename)

    # Run single instance
    # start_time = time.time()
    # best_tour, best_distance, generation = genetic_algorithm(
    #     pop_size=400, generations=2100, crossover_rate=0.87, mutation_rate=0.14
    # )
    # elapsed_time = round((time.time() - start_time), 3)
    # results = [(400, 0.14, 0.87, best_distance, elapsed_time, generation, best_tour)]
    # save_to_csv("results/single_" + filename + "_results.csv", results)
    # show_best_result()