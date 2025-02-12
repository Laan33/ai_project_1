import random

POPULATION_SIZE = 500
GENOME_SIZE = 70
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.84
GENERATIONS = 100


def random_genome():
    return [random.choice([0, 1]) for _ in range(GENOME_SIZE)]

def init_population(population_size, genome_size):
    return [random_genome() for _ in range(population_size)]

def fitness(genome):
    return sum(genome)

def select_parent(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual

def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(0, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]
    else:
        return parent1, parent2

def mutate(genome):
    for i in range(len(genome)):
        if random.random() < MUTATION_RATE:
            genome[i] = abs(genome[i] - 1)
    return genome

def genetic_algorithm():
    population = init_population(POPULATION_SIZE, GENOME_SIZE)

    for generation in range(GENERATIONS):
        fitness_values = [fitness(genome) for genome in population]

        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = select_parent(population, fitness_values)
            parent2 = select_parent(population, fitness_values)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])

        population = new_population

        fitness_values = [fitness(genome) for genome in population]
        best_fitness = max(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_index]
    print(f'Best Solution: {best_solution}')
    print(f'Best Fitness: {fitness(best_solution)}')

if __name__ == '__main__':
    genetic_algorithm()


