import numpy as np

from dipl.data.search_space import get_objective_function_and_params


def abc(signal: list,
        time: list,
        objective_function_type: str,
        ):
    """
    Artificial Bee Colony algorithm.
    :param objective_function: The objective function to minimize.
    :param bounds: The bounds of the search space.
    :param n_bees: The number of bees in the colony.
    :param max_iter: The maximum number of iterations.
    :param max_trials: The maximum number o f trials before a food source is abandoned.
    :return: The best solution found by the algorithm.
    """
    param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)

    n_bees = 30
    max_iter = 100
    max_trials = 10
    n_dim = len(param_ranges)
    lb = [b[0] for b in param_ranges]
    ub = [b[1] for b in param_ranges]

    population = np.random.uniform(lb, ub, (n_bees, n_dim))

    # evaluate the population
    fitness = np.apply_along_axis(objective_function, 1, population, signal)

    # initialize the trial counter
    trial_counter = np.zeros(n_bees)

    # find the best solution
    best_index = np.argmin(fitness)
    best_solution = population[best_index]

    # main loop
    for _ in range(max_iter):
        # employed bees phase
        for i in range(n_bees):
            # generate a new solution
            new_solution = population[i].copy()
            j = np.random.randint(n_dim)
            k = np.random.randint(n_bees)
            while k == i:
                k = np.random.randint(n_bees)
            phi = np.random.uniform(-1, 1)
            new_solution[j] += phi * (population[i][j] - population[k][j])
            new_solution = np.clip(new_solution, lb, ub)

            # evaluate the new solution
            new_fitness = objective_function(new_solution, signal)

            # greedy selection
            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness
                trial_counter[i] = 0
            else:
                trial_counter[i] += 1

        # onlooker bees phase
        for i in range(n_bees):
            # select a food source
            p = fitness / fitness.sum()
            j = np.random.choice(n_bees, p=p)

            # generate a new solution
            new_solution = population[j].copy()
            k = np.random.randint(n_dim)
            l = np.random.randint(n_bees)
            while l == j:
                l = np.random.randint(n_bees)
            phi = np.random.uniform(-1, 1)
            new_solution[k] += phi * (population[j][k] - population[l][k])
            new_solution = np.clip(new_solution, lb, ub)

            # evaluate the new solution
            new_fitness = objective_function(new_solution, signal)

            # greedy selection
            if new_fitness < fitness[j]:
                population[j] = new_solution
                fitness[j] = new_fitness
                trial_counter[j] = 0
            else:
                trial_counter[j] += 1

        # scout bees phase
        for i in range(n_bees):
            if trial_counter[i] > max_trials:
                population[i] = np.random.uniform(lb, ub, n_dim)
                fitness[i] = objective_function(population[i], signal)
                trial_counter[i] = 0

        # find the best solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < objective_function(best_solution, signal):
            best_solution = population[best_index]

    return best_solution

# import numpy as np
# from scipy.optimize import rosen
#
# from dipl.data.search_space import get_objective_function_and_params


# def abc(objective_function, lb, ub, colony_size=30, n_iter=5000,
#         max_trials=100):

# def abc(
#         signal: list,
#         time: list,
#         objective_function_type: str,
# ):
#     colony_size = 30
#     n_iter = 5000
#     max_trials = 100
#     param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)
#     """
#     Artificial Bee Colony Algorithm.
#
#     Parameters
#     ----------
#     objective_function : callable
#         The function to minimize.
#     lb : array_like
#         The lower bounds of the design variables.
#     ub : array_like
#         The upper bounds of the design variables.
#     colony_size : int, optional
#         The number of bees in the colony. Default is 30.
#     n_iter : int, optional
#         The number of iterations to perform. Default is 5000.
#     max_trials : int, optional
#         The maximum number of trials without any improvement before a food source is abandoned. Default is 100.
#
#     Returns
#     -------
#     best_solution : ndarray
#         The best solution found by the algorithm.
#     best_val : float
#         The value of the objective function at the best solution.
#     """
#     # Check input
#     # lb = np.asarray(lb)
#     # ub = np.asarray(ub)
#     lb = [p[0] for p in param_ranges],
#     ub = [p[1] for p in param_ranges],
#
#     lb = np.array(lb[0])
#     ub = np.array(ub[0])
#
#     # assert len(lb) == len(ub), 'Lower- and upper-bounds must be of same length.'
#     # assert np.all(ub > lb), 'All upper-bound values must be greater than lower-bound values.'
#
#     n_params = len(lb)
#     half_colony_size = colony_size // 2
#
#     # Initialize population
#     population = np.random.rand(colony_size, n_params)
#     population *= (ub - lb)
#     population += lb
#
#     # Evaluate population
#     fitness = np.apply_along_axis(objective_function, 1, population, signal)
#
#     # Determine best solution
#     min_idx = np.argmin(fitness)
#     best_solution = population[min_idx].copy()
#     best_val = fitness[min_idx]
#
#     # Set trial counters to zero
#     trials = np.zeros(colony_size)
#
#     # Main loop
#     for _ in range(n_iter):
#         # Employed bees phase
#         for i in range(half_colony_size):
#             # Choose a parameter to modify
#             param_to_modify = np.random.randint(n_params)
#
#             # Choose a bee from the colony
#             bee_ix = i if np.random.rand() < 0.5 else half_colony_size + i
#
#             # Generate a new solution
#             new_solution = population[i].copy()
#             new_solution[param_to_modify] += np.random.uniform(-1, 1) * (
#                         population[i][param_to_modify] - population[bee_ix][param_to_modify])
#
#             # Make sure the new solution is within the bounds
#             new_solution[param_to_modify] = np.clip(new_solution[param_to_modify], lb[param_to_modify],
#                                                     ub[param_to_modify])
#
#             # Evaluate new solution
#             new_val = objective_function(new_solution, signal)
#
#             # Greedy selection
#             if new_val < fitness[i]:
#                 population[i] = new_solution.copy()
#                 fitness[i] = new_val
#                 trials[i] = 0
#             else:
#                 trials[i] += 1
#
#         # Onlooker bees phase
#         for i in range(half_colony_size):
#             # Choose a food source using roulette wheel selection based on fitness values
#             probs = (np.max(fitness) - fitness) / (np.max(fitness) - np.min(fitness))
#             probs /= probs.sum()
#             food_source_ix = np.random.choice(colony_size, p=probs)
#
#             # Choose a parameter to modify
#             param_to_modify = np.random.randint(n_params)
#
#             # Choose a bee from the colony (not the same as the food source)
#             bee_ix = food_source_ix if np.random.rand() < 0.5 else half_colony_size + food_source_ix
#
#             # Generate a new solution
#             new_solution = population[food_source_ix].copy()
#             new_solution[param_to_modify] += np.random.uniform(-1, 1) * (
#                         population[food_source_ix][param_to_modify] - population[bee_ix][param_to_modify])
#
#             # Make sure the new solution is within the bounds
#             new_solution[param_to_modify] = np.clip(new_solution[param_to_modify], lb[param_to_modify],
#                                                     ub[param_to_modify])
#
#             # Evaluate new solution
#             new_val = objective_function(new_solution, signal)
#
#             # Greedy selection
#             if new_val < fitness[food_source_ix]:
#                 population[food_source_ix] = new_solution.copy()
#                 fitness[food_source_ix] = new_val
#                 trials[food_source_ix] = 0
#             else:
#                 trials[food_source_ix] += 1
#
#             # Scout bees phase
#         for i in range(colony_size):
#             if trials[i] >= max_trials:
#                 # Abandon food source
#                 population[i] = np.random.rand(n_params) * (ub - lb) + lb
#                 fitness[i] = objective_function(population[i])
#                 trials[i] = 0
#
#             # Update best solution
#         min_idx = np.argmin(fitness)
#         if fitness[min_idx] < best_val:
#             best_solution = population[min_idx].copy()
#             best_val = fitness[min_idx]
#
#     return best_solution, best_val

