import numpy as np
import random

# Parametre portfólia
NUM_STOCKS = 10  # Počet akcií
POPULATION_SIZE = 100  # Veľkosť populácie
GENERATIONS = 10  # Počet generácií
MUTATION_RATE = 0.1  # Miera mutácie

# Generovanie náhodných ročných výnosov a rizík (volatilita)
np.random.seed(42)
expected_returns = np.random.uniform(0.05, 0.50, NUM_STOCKS)  # Očakávané ročné výnosy
risks = np.random.uniform(0.1, 0.5, NUM_STOCKS)  # Očakávané riziká (volatilita)

# Definovanie fitness funkcie
def fitness_function(weights):
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.diag(risks**2), weights)))
    return portfolio_return - portfolio_risk

# Normalizácia váh
def normalize_weights(weights):
    return weights / np.sum(weights)

# Generovanie počiatočnej populácie
def generate_population(size):
    return [normalize_weights(np.random.rand(NUM_STOCKS)) for _ in range(size)]

# Turnajová selekcia
def tournament_selection(population):
    tournament = random.sample(population, 5)
    return max(tournament, key=fitness_function)

# Ruletový výber
def roulette_selection(population):
    fitness_values = np.array([fitness_function(ind) for ind in population])
    fitness_sum = np.sum(fitness_values)
    selection_probs = fitness_values / fitness_sum
    return population[np.random.choice(len(population), p=selection_probs)]

# Výber podľa poradia (Rank Selection)
def rank_selection(population):
    fitness_values = np.array([fitness_function(ind) for ind in population])
    ranked_indices = np.argsort(-fitness_values)
    selection_probs = 1 / (np.arange(len(population)) + 1)
    selection_probs /= np.sum(selection_probs)
    return population[np.random.choice(len(population), p=selection_probs)]

# Hlavná funkcia genetického algoritmu
def genetic_algorithm(selection_method):
    population = generate_population(POPULATION_SIZE)
    
    for generation in range(GENERATIONS):
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1 = selection_method(population)
            parent2 = selection_method(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))
        
        population = new_population
        
        best_solution = max(population, key=fitness_function)
        yield generation, best_solution, fitness_function(best_solution)

# Kríženie
def crossover(parent1, parent2):
    point = random.randint(1, NUM_STOCKS - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return normalize_weights(child1), normalize_weights(child2)

# Mutácia
def mutate(individual):
    if random.random() < MUTATION_RATE:
        idx = random.randint(0, NUM_STOCKS - 1)
        individual[idx] = random.uniform(0.0, 1.0)
        return normalize_weights(individual)
    return individual

# Porovnanie selekcií
def compare_selections():
    methods = {
        'Turnajová selekcia': tournament_selection,
        'Ruletový výber': roulette_selection,
        'Výber podľa poradia': rank_selection
    }

    results = {}
    
    for method_name, method in methods.items():
        print(f"\n{method_name}:\n" + "=" * 30)
        best_portfolio = None
        best_fitness = float('-inf')
        for generation, solution, fitness in genetic_algorithm(method):
            print(f"Generácia {generation + 1}: Najlepšie váhy: {solution}, Výnos: {fitness:.4f}")
            if fitness > best_fitness:
                best_fitness = fitness
                best_portfolio = solution
        
        results[method_name] = (best_portfolio, best_fitness)

    print("\nNajlepšie výsledky:")
    for method, (portfolio, fitness) in results.items():
        print(f"{method}: Výnos: {fitness:.4f}, Váhy: {portfolio}")

# Spustenie porovnania selekcií
compare_selections()
