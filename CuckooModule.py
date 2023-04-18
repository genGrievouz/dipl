import numpy as np
import math
import matplotlib.pyplot as plt

from dipl.functions.ldrw import LDRW
from dipl.functions.load_data import load_data, remove_negative_values, preprocessing, normalize


def get_params_for_cuckoo():
    def initialize_nests(pop_size, param_range):
        nests = []
        for i in range(pop_size):
            alpha = np.random.uniform(param_range[0], param_range[1])
            beta = np.random.uniform(param_range[2], param_range[3])
            nests.append([alpha, beta])
        return nests

    # Evaluate fitness of population
    def evaluate_fitness(nests, x, auc):
        fitness = []
        for nest in nests:
            alpha = nest[0]
            beta = nest[1]
            y = [(auc * (i ** alpha) * np.exp(-1 * i / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for i in
                 x]
            fitness.append(sum(y))
        return fitness

    # Abandon solutions
    def abandon_solutions(nests, num_abandoned):
        # Randomly select nests to abandon
        param_range = [0.1, 10.0, 0.1, 10.0]
        abandoned_indices = np.random.choice(len(nests), size=num_abandoned, replace=False)
        # Replace abandoned nests with new random solutions
        for i in abandoned_indices:
            nests[i][0] = np.random.uniform(param_range[0], param_range[1])
            nests[i][1] = np.random.uniform(param_range[2], param_range[3])
        return nests

    # Generate new solutions using Levy flights
    def levy_flight(nests, param_range, beta=1.5):
        new_nests = []
        for nest in nests:
            alpha = nest[0]
            beta = nest[1]
            # Generate step size from Levy distribution
            step_size = np.random.standard_cauchy(size=2)
            step_size = step_size / abs(step_size) ** (1 / beta)
            # Generate new nest
            new_nest = nest + step_size * 0.01 * (param_range[1] - param_range[0])
            # Bound parameters within range
            new_nest[0] = max(param_range[0], min(new_nest[0], param_range[1]))
            new_nest[1] = max(param_range[2], min(new_nest[1], param_range[3]))
            new_nests.append(new_nest)
        return new_nests

    def update_population(nests, new_nests, fitness, new_fitness, num_keep):
        # Combine old and new nests and fitness values
        all_nests = nests + new_nests
        all_fitness = fitness + new_fitness
        # Sort nests by fitness
        sorted_indices = np.argsort(all_fitness)[::-1]
        sorted_nests = [all_nests[i] for i in sorted_indices]
        # Keep best nests and discard rest
        return sorted_nests[:num_keep]

    def cuckoo_search_gamma(x, auc, pop_size=10, num_iterations=50, num_abandoned=3, num_keep=10, beta=1.5):
        # Initialize population
        param_range = [0.1, 10.0, 0.1, 10.0]
        nests = initialize_nests(pop_size, param_range)

        # Evaluate fitness of initial population
        fitness = evaluate_fitness(nests, x, auc)

        # Main loop
        for i in range(num_iterations):
            # Generate new solutions using Levy flights
            new_nests = levy_flight(nests, param_range, beta)

            # Evaluate fitness of new solutions
            new_fitness = evaluate_fitness(new_nests, x, auc)

            # Abandon solutions
            nests = abandon_solutions(nests, num_abandoned)

            # Update population
            nests = update_population(nests, new_nests, fitness, new_fitness, num_keep)

            # Update fitness
            fitness = evaluate_fitness(nests, x, auc)

            # Print best fitness
            best_fitness = max(fitness)
            print(f"Iteration {i + 1}: Best fitness = {best_fitness}")
            # Get best solution

        best_index = np.argmax(fitness)
        best_params = nests[best_index]
        alpha, beta = best_params

        # Return best solution
        return alpha, beta

    auc = 0.9
    data = load_data()
    k = 'exp13_roivelke_inp_tis_111007'
    x = normalize(remove_negative_values(preprocessing((data[k]['signal']))))
    threshold = 0.03
    removed_indices = [index for index in range(len(x)) if x[index] < threshold]
    x = [i for i in x if i > threshold]
    time = data[k]['time']
    time = [time[i] for i in range(len(time)) if i not in removed_indices]
    alpha, beta = cuckoo_search_gamma(x, auc, pop_size=10, num_iterations=50, num_abandoned=3, num_keep=10, beta=1.5)

    ldrw = LDRW(x, time)
    fit_ldrw = ldrw.fit



    plt.figure()
    plt.plot(time, x, 'o', label='data')
    # plt.plot(time, fit_lognormal, '-', label='lognormal')
    # plt.plot(time, fit_gamma, '-', label='gamma variate')
    # plt.plot(time, fit_ldrw, '-', label='ldrw')
    # plt.plot(time, fit_fpt, '-', label='fpt')
    # plt.plot(time, fit_lagged, '-', label='lagged normal')
    plt.title(k)
    plt.legend(loc='upper right')
    plt.ylabel('Intensity')
    plt.xlabel('Time [s]')
    plt.show()


