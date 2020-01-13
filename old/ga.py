import numpy as np
import pandas as pd
import random

def initial(dv, pop_size):
    pop = [list(np.random.permutation(dv)) for i in range(pop_size)]
    return pop

def fitness_aux(x, data):
    y=[data[x[i]][x[i+1]] for i in range(len(x)) if i < (len(x)-1)]
    y.append(data[x[-1]][x[0]])
    return sum(y)

def fitness_function(x, data):
    y = [fitness_aux(x[i], data) for i in range(len(x))]
    return y

def tournament_selection(x, fitness, size = 5):
    candidates = list(np.random.choice(range(len(x)), size))
    candidates_fitness = [fitness[i] for i in candidates]
    champion = candidates[candidates_fitness.index(min(candidates_fitness))]
    return champion

def select_parents(x, fitness):
    parents_index = [tournament_selection(x, fitness) for i in range(len(x))]
    parents = [x[i] for i in parents_index]
    return parents

def order_crossover(a,b):
    points = list(np.random.choice(range(1,len(a)), 2, replace=False))
    points.sort()
    point1, point2 = points[0], points[1]
    
    child_a = [-1]*len(a)
    child_a[point1 : point2] = a[point1 : point2]
    conflict_a = a[point1 : point2]
    mapping_a = [i for i in b[point2:] + b[:point2] if i not in conflict_a] 
    diff=len(a)-point2
    child_a[point2:] = mapping_a[:diff]
    child_a[:point1] = mapping_a[diff:]
    
    child_b = [-1]*len(b)
    child_b[point1 : point2] = b[point1 : point2]
    conflict_b = b[point1 : point2]
    mapping_b = [i for i in a[point2:] + a[:point2] if i not in conflict_b] 
    diff=len(b)-point2
    child_b[point2:] = mapping_b[:diff]
    child_b[:point1] = mapping_b[diff:]
    
    return child_a, child_b

def inversion_mutation(a):
    points = list(np.random.choice(range(1,len(a)), 2, replace=False))
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

