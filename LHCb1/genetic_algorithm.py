from __future__ import print_function
from __future__ import division
import pygad
import concurrent.futures

import numpy as np
import random
from random import randint
import operator
from array import array
import math
import time
import itertools
import sys
import os
import csv

import pandas as pd
import matplotlib.pyplot as plt
import functools
import warnings
from subprocess import check_output

from ipywidgets import interact
from root_pandas import read_root
from scipy import stats as st
import seaborn as sns

# Constants
NUM_BITS = 4
POP_SIZE = 12
NUM_GENERATIONS = 1000
CXPB = 0.6  # Probability of crossover
MUTPB = 0.02  # Probability of mutation
NUM_PARAMETERS = 6

# Probability ranges
PROB_MIN = 0.2
PROB_MAX = 0.9

# Delete the values.csv if it exists
if os.path.exists('values_4bit.csv'):
    os.remove('values_4bit.csv')

# Create the values_4bit.csv and write the header
with open('values_4bit.csv', 'w') as file:
    file.write('H1_ProbPi,H2_ProbPi,H3_ProbPi,H1_ProbK,H2_ProbK,H3_ProbK,Fitness\n')

# Fitness function
def fitness_function(ga_instance, solution, solution_idx):
    # Convert the solution into the preselection string
    preselection = ( f"H1_ProbPi < {solution[0]} & H2_ProbPi < {solution[1]} & H3_ProbPi < {solution[2]} & H1_ProbK > {solution[3]} & H2_ProbK > {solution[4]} & H3_ProbK > {solution[5]} & !H1_isMuon & !H2_isMuon & !H3_isMuon"
    )

    # Function to calculate significance
    M_K = 493.677  # MeV/c²
    real_data = read_root(
        ['/data/B2HHH_MagnetUp.root', '/data/B2HHH_MagnetDown.root'], 
        where=preselection  # Using the dynamically generated preselection
    )

    # Calculating properties for H1, H2, H3 particles
    for i in range(1, 4):
        real_data[f'H{i}_P'] = np.sqrt(real_data[f'H{i}_PX']**2 + real_data[f'H{i}_PY']**2 + real_data[f'H{i}_PZ']**2)
        real_data[f"H{i}_E"] = np.sqrt(real_data[f'H{i}_P']**2 + M_K**2)

    # Calculate properties of the B meson
    real_data['B_E'] = real_data['H1_E'] + real_data['H2_E'] + real_data['H3_E']
    real_data['B_PX'] = real_data['H1_PX'] + real_data['H2_PX'] + real_data['H3_PX']
    real_data['B_PY'] = real_data['H1_PY'] + real_data['H2_PY'] + real_data['H3_PY']
    real_data['B_PZ'] = real_data['H1_PZ'] + real_data['H2_PZ'] + real_data['H3_PZ']
    real_data['B_P'] = np.sqrt(real_data['B_PX']**2 + real_data['B_PY']**2 + real_data['B_PZ']**2)
    real_data['B_M'] = np.sqrt(real_data['B_E']**2 - real_data['B_P']**2)
    real_data['B_CH'] = real_data['H1_Charge'] + real_data['H2_Charge'] + real_data['H3_Charge']


    print("Lunghezza:", len(real_data))
    N_P = np.sum((real_data['B_CH'] + 1) / 2)
    N_M = -np.sum((real_data['B_CH'] - 1) / 2)

    print("Numeratore", (N_P - N_M), 'Denominatore', (N_P + N_M), 'A', (N_P - N_M) / (N_P + N_M), 'S', np.sqrt((1 - ((N_P - N_M) / (N_P + N_M))**2) / (N_P + N_M)))
    
    if (N_P + N_M) == 0:
        return -0.1
    
    if (N_P - N_M) / (N_P + N_M) == 0:
        return -0.1

    A = np.abs((N_P - N_M) / (N_P + N_M))
    s_A = np.sqrt((1 - A**2) / (N_P + N_M))
    significance = A / s_A

    if significance == float('inf') or significance == float('-inf'):
        return -0.1

    return -significance

# Gene space for each parameter
gene_space = [{'low': PROB_MIN, 'high': PROB_MAX} for _ in range(NUM_PARAMETERS)]

# Function to save the population and fitness to CSV
def on_generation(ga_instance):
    population = ga_instance.population
    fitness = ga_instance.last_generation_fitness
    with open('values_4bit.csv', 'a') as file:
        for sol, fit in zip(population, fitness):
            file.write(','.join(map(str, sol)) + ',' + str(fit) + '\n')

# Create an instance of the GA class
ga_instance = pygad.GA(num_generations=NUM_GENERATIONS,
                       num_parents_mating=int(CXPB * POP_SIZE),
                       fitness_func=fitness_function,
                       sol_per_pop=POP_SIZE,
                       num_genes=NUM_PARAMETERS,
                       gene_space=gene_space,
                       parent_selection_type="rws",
                       keep_parents=2,
                       crossover_type="single_point",
                       mutation_type="random",
                       mutation_probability=MUTPB,
                       on_generation=on_generation)

# Run the GA to optimize the parameters
ga_instance.run()

# After the generations complete, print the best solution
best_solution, best_solution_fitness, _ = ga_instance.best_solution()
print("Best Solution:", best_solution)
print("Best Solution Fitness:", best_solution_fitness)

# Plotting the fitness over generations
ga_instance.plot_fitness()
