import numpy as np
import pandas as pd
import random

def initial(dv, pop_size):
    pop = [list(np.random.permutation(dv)) for i in range(pop_size)]
    return pop


def fitness_aux(x, data):
    y = [data[x[i]][x[i + 1]] for i in range(len(x)) if i < (len(x) - 1)]
    y.append(data[x[-1]][x[0]])
    return sum(y)


def fitness_function(x, data):
    y = [fitness_aux(x[i], data) for i in range(len(x))]
    return y


def tournament_selection(x, fitness, size=5):
    candidates = list(np.random.choice(range(len(x)), size))
    candidates_fitness = [fitness[i] for i in candidates]
    champion = candidates[candidates_fitness.index(min(candidates_fitness))]
    return champion


def select_parents(x, fitness):
    parents_index = [tournament_selection(x, fitness) for i in range(len(x))]
    parents = [x[i] for i in parents_index]
    return parents


def order_crossover(a, b):
    points = list(np.random.choice(range(1, len(a)), 2, replace=False))
    points.sort()
    point1, point2 = points[0], points[1]

    child_a = [-1] * len(a)
    child_a[point1: point2] = a[point1: point2]
    conflict_a = a[point1: point2]
    mapping_a = [i for i in b[point2:] + b[:point2] if i not in conflict_a]
    diff = len(a) - point2
    child_a[point2:] = mapping_a[:diff]
    child_a[:point1] = mapping_a[diff:]

    child_b = [-1] * len(b)
    child_b[point1: point2] = b[point1: point2]
    conflict_b = b[point1: point2]
    mapping_b = [i for i in a[point2:] + a[:point2] if i not in conflict_b]
    diff = len(b) - point2
    child_b[point2:] = mapping_b[:diff]
    child_b[:point1] = mapping_b[diff:]

    return child_a, child_b


def inversion_mutation(a):
    points = list(np.random.choice(range(1, len(a)), 2, replace=False))
    points.sort()
    point1, point2 = points[0], points[1]
    child = a[:point1] + a[point1:point2][::-1] + a[point2:]
    return child


def elitism_replacement(population, fitness, offspring, fitness_offspring):
    if min(fitness) < min(fitness_offspring):
        offspring[fitness_offspring.index(max(fitness_offspring))] = population[fitness.index(min(fitness))]
    return offspring


def save_best_fitness(population, fitness):
    best = fitness.index(min(fitness))
    return population[best], fitness[best]


def ga_search(data, num_gen=1000, prob_cross=0.6, prob_mut=0.6):
    decision_variables = list(range(len(data)))
    population = initial(decision_variables, 20)
    fitness = fitness_function(population, data)
    best = save_best_fitness(population, fitness)
    generation, best_fitness, fittest = [0], [best[1]], [str(best[0])]

    for gen in range(num_gen):
        parents = select_parents(population, fitness)
        offspring = parents.copy()
        for i in range(0, len(population), 2):
            if (np.random.uniform() < prob_cross):
                offspring[i], offspring[i + 1] = order_crossover(parents[i], parents[i + 1])
        for i in range(len(population)):
            if (np.random.uniform() < prob_mut):
                offspring[i] = inversion_mutation(offspring[i])
        fitness_offspring = fitness_function(offspring, data)
        population = elitism_replacement(population, fitness, offspring, fitness_offspring)
        fitness = fitness_function(population, data)
        best = save_best_fitness(population, fitness)
        generation.append(gen + 1), best_fitness.append(best[1]), fittest.append(str(best[0]))

    generation = pd.Series(generation)
    best_fitness = pd.Series(best_fitness)
    fittest = pd.Series(fittest)
    run = pd.concat([generation, best_fitness, fittest], axis=1)
    run.columns = ['Generation', 'Fitness', 'Fittest']
    run.drop_duplicates('Fittest', inplace=True)

    return run