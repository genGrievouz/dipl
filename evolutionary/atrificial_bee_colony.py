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

    n_bees = 15
    max_iter = 100
    max_trials = 10
    n_dim = len(param_ranges)
    lb = [b[0] for b in param_ranges]
    ub = [b[1] for b in param_ranges]
    # initialize the population
    population = np.random.uniform(lb, ub, (n_bees, n_dim))

    # evaluate the population
    fitness = np.apply_along_axis(objective_function, 1, population)

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
            new_fitness = objective_function(new_solution)

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
            new_fitness = objective_function(new_solution)

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
                fitness[i] = objective_function(population[i])
                trial_counter[i] = 0

        # find the best solution
        best_index = np.argmin(fitness)
        if fitness[best_index] < objective_function(best_solution):
            best_solution = population[best_index]

    return best_solution
