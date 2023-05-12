# import random
# import numpy as np
#
# from dipl.data.search_space import get_objective_function_and_params
#
#
# def ant_colony_optimization(
#         signal: list,
#         time: list,
#         objective_function_type: str,
#         num_ants=10,
#         num_iterations=100,
#         evaporation_rate=0.1,
#         alpha=1, beta=2
# ):
#
#     param_range, objective_function = get_objective_function_and_params(signal,
#                                                                         time,
#                                                                         objective_function_type
#                                                                         )
#
#     """
#     Ant Colony Optimization algorithm for finding the optimal parameters for a given objective function.
#
#     Parameters:
#     signal (list): The signal to be modeled.
#     time (list): The time points corresponding to the signal.
#     objective_function (function): The objective function to optimize.
#     num_ants (int): The number of ants in the colony.
#     num_iterations (int): The number of iterations to run the algorithm.
#     evaporation_rate (float): The rate at which pheromone evaporates.
#     alpha (float): The importance of the pheromone trail in choosing the next parameter.
#     beta (float): The importance of the fitness value in choosing the next parameter.
#
#     Returns:
#     best_params (dict): The best parameters found by the algorithm.
#     """
#
#     num_params = len(param_range)
#     pheromone = np.ones(num_params) / num_params
#
#     # Initialize the best parameters and fitness value
#     best_params = {}
#     best_fitness = float('-inf')
#
#     # Initialize the colony of ants
#     ants = []
#     for i in range(num_ants):
#         ant_params = {}
#         for j in range(num_params):
#             param_min, param_max = param_range[j]
#             ant_params[j] = random.uniform(param_min, param_max)
#         ants.append(ant_params)
#
#     # Run the algorithm for the specified number of iterations
#     for i in range(num_iterations):
#         fitness_values = []
#         for ant_params in ants:
#             fitness = objective_function(ant_params, signal)
#             fitness_values.append(fitness)
#
#             # Update the best parameters and fitness value
#             if fitness > best_fitness:
#                 best_params = ant_params
#                 best_fitness = fitness
#
#         # Update the pheromone trail
#         pheromone *= (1 - evaporation_rate)
#         for ant_params, fitness in zip(ants, fitness_values):
#             for j in range(num_params):
#                 param_value = ant_params[j]
#                 pheromone[j] += evaporation_rate * fitness / objective_function(ant_params, signal) * (
#                                         param_value == best_params[j])
#
#         # Choose the next set of candidate parameters
#         for ant_params in ants:
#             candidate_params = {}
#             for j in range(num_params):
#                 param_min, param_max = param_range[j]
#                 param_min = int(param_min)
#                 param_max = int(param_max)
#                 param_probs = np.zeros(param_max - param_min + 1)
#                 for k in range(param_min, param_max + 1):
#                     param_probs[k - param_min] = pheromone[j] ** alpha * (1 / abs(ant_params[j] - k)) ** beta
#                 param_probs_sum = param_probs.sum()
#                 candidate_params[str(j)] = np.random.choice(np.arange(param_min, param_max + 1),
#                                                             p=param_probs / param_probs_sum)
#             ant_params.update(candidate_params)


import numpy as np
from scipy.optimize import minimize

from dipl.data.search_space import get_objective_function_and_params


# Define the ant colony algorithm
def ant_colony_optimization(signal,
                         time,
                         objective_function_type,
                         n_ants=50,
                         n_iterations=100,
                         alpha=1,
                         beta=1,
                         evaporation_rate=0.5,
                         Q=100,
                         max_tau=2
                         ):

    # Define the objective function and parameter ranges
    param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)

    # Define the size
    size = len(param_ranges)

    # Initialize the pheromone trail
    tau = np.ones((size, len(signal))) * max_tau

    # Initialize the best solution and its objective function value
    best_solution = None
    best_value = np.inf

    # Start the iterations
    for iteration in range(n_iterations):
        # Initialize the ants' solutions and their objective function values
        solutions = np.zeros((n_ants, 3))
        values = np.zeros(n_ants)

        # Move the ants
        for ant in range(n_ants):
            # Choose the next parameter using the pheromone trail and the heuristic information
            probs = (tau[:, :, np.newaxis] ** alpha) * ((1 / (values[:, np.newaxis] + 1e-10)) ** beta)
            probs /= np.sum(probs, axis=0)
            # param_indices = np.argmax(np.random.multinomial(1, probs[:, :, 0].T), axis=1)
            # params = np.array(
            #     [auc_choices[param_indices[0]], mean_choices[param_indices[1]], std_choices[param_indices[2]]])
            params = param_ranges[np.random.choice(np.arange(0, size), p=probs[:, :, 0].T)]
            solutions[ant, :] = params

            # Calculate the objective function value of the solution
            values[ant] = objective_function(params, signal)

        # Update the pheromone trail
        delta_tau = np.zeros((3, signal.shape[0]))
        for ant in range(n_ants):
            for param_index, param_value in enumerate(solutions[ant, :]):
                delta_tau[param_index, int(param_value)] += Q / values[ant]
        tau = (1 - evaporation_rate) * tau + delta_tau

        # Update the best solution and its objective function value
        current_best_index = np.argmin(values)
        if values[current_best_index] < best_value:
            best_solution = solutions[current_best_index, :]
            best_value = values[current_best_index]

    return best_solution, best_value
