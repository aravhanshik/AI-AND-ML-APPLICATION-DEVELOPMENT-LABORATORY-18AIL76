import random

# Define the knapsack problem parameters
max_weight = 50
items = [
{'name': 'item1', 'weight': 10, 'value': 60},
{'name': 'item2', 'weight': 20, 'value': 100},
{'name': 'item3', 'weight': 30, 'value': 120},
{'name': 'item4', 'weight': 15, 'value': 50},
{'name': 'item5', 'weight': 25, 'value': 60},
]

# Genetic algorithm parameters
population_size = 100
generations = 50
mutation_rate = 0.1

# Initialize a population of random solutions
def initialize_population(pop_size, item_count):
    population = []
    for _ in range(pop_size):
        solution = [random.randint(0, 1) for _ in range(item_count)]
        population.append(solution)
    return population
  
# Calculate the fitness of a solution
def fitness(solution):
    total_weight = sum(item['weight'] for item, bit in zip(items, solution) if bit)
    total_value = sum(item['value'] for item, bit in zip(items, solution) if bit)
    return total_value if total_weight <= max_weight else 0
  
# Perform single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2
  
# Mutate a solution
def mutate(solution, mutation_rate):
    mutated_solution = []
    for bit in solution:
        if random.random() < mutation_rate:
            mutated_solution.append(1 - bit) # Flip the bit with a probability of mutation_rate
        else:
            mutated_solution.append(bit)
    return mutated_solution
  
# Main genetic algorithm loop
population = initialize_population(population_size, len(items))
for generation in range(generations):
    population = sorted(population, key=lambda x: fitness(x), reverse=True)
    new_population = population[:population_size // 2]
    for _ in range(population_size // 2):
        parent1, parent2 = random.choices(population, k=2)
        child1, child2 = crossover(parent1, parent2)
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        new_population.extend([child1, child2])
    population = new_population
  
best_solution = max(population, key=fitness)
best_value = sum(item['value'] for item, bit in zip(items, best_solution) if bit)
best_weight = sum(item['weight'] for item, bit in zip(items, best_solution) if bit)
print("Best solution:", best_solution)
print("Total value:", best_value)
print("Total weight:", best_weight)
