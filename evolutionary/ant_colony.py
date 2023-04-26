import random
import numpy as np
import math


def ant_colony_optimization(
        signal: list,
        time: list,
        ts: float,
        param_range,
        num_ants=10,
        num_iterations=100,
        evaporation_rate=0.1,
        alpha=1, beta=2
):
    def objective_function(params):
        auc, alpha, beta = params
        output = [(auc * (x ** alpha) * np.exp(-1 * x / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for x in signal]
        return np.sum(np.abs(output))

    """
    Ant Colony Optimization algorithm for finding the optimal parameters for a given objective function.

    Parameters:
    obj_func (function): The objective function to optimize.
    param_range (list of tuples): The range of candidate parameters for the model.
    num_ants (int): The number of ants in the colony.
    num_iterations (int): The number of iterations to run the algorithm.
    evaporation_rate (float): The rate at which pheromone evaporates.
    alpha (float): The importance of the pheromone trail in choosing the next parameter.
    beta (float): The importance of the fitness value in choosing the next parameter.

    Returns:
    best_params (dict): The best parameters found by the algorithm.
    """

    # Initialize the pheromone trail
    num_params = len(param_range)
    pheromone = np.ones(num_params) / num_params

    # Initialize the best parameters and fitness value
    best_params = {}
    best_fitness = float('-inf')

    # Initialize the colony of ants
    ants = []
    for i in range(num_ants):
        ant_params = {}
        for j in range(num_params):
            param_min, param_max = param_range[j]
            ant_params[j] = random.uniform(param_min, param_max)
        ants.append(ant_params)

    # Run the algorithm for the specified number of iterations
    for i in range(num_iterations):
        # Evaluate the fitness of each ant's parameters
        fitness_values = []
        for ant_params in ants:
            print(ant_params)
            fitness = objective_function(**ant_params)
            fitness_values.append(fitness)

            # Update the best parameters and fitness value
            if fitness > best_fitness:
                best_params = ant_params
                best_fitness = fitness

        # Update the pheromone trail
        pheromone *= (1 - evaporation_rate)
        for ant_params, fitness in zip(ants, fitness_values):
            for j in range(num_params):
                param_value = ant_params[j]
                pheromone[j] += evaporation_rate * fitness / objective_function(**best_params) * (param_value == best_params[j])

        # Choose the next set of candidate parameters
        for ant_params in ants:
            candidate_params = {}
            for j in range(num_params):
                param_min, param_max = param_range[j]
                param_probs = np.zeros(param_max - param_min + 1)
                for k in range(param_min, param_max + 1):
                    param_probs[k - param_min] = pheromone[j] ** alpha * (1 / abs(ant_params[j] - k)) ** beta
                param_probs /= param_probs.sum()
                candidate_params[j] = np.random.choice(np.arange(param_min, param_max + 1), p=param_probs)
            ant_params.update(candidate_params)

    return best_params
