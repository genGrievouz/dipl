import numpy as np


def cuckoo_search(objective_function, dimension, lb, ub, pa=0.25, nest_count=25, max_iter=100):
    """
    lb - lower bound
    ub - upper bound
    pa - probability of abandoning a nest
    nest_count - number of nests
    max_iter - number of iterations
    """
    # Generate initial solutions
    nests = np.random.uniform(low=lb, high=ub, size=(nest_count, dimension))
    fitness = np.apply_along_axis(objective_function, 1, nests)

    # Find the best nest
    fmin, best_nest = min(zip(fitness, nests))

    # Start iterations
    for _ in range(max_iter):
        # Generate new solutions by cuckoo
        new_nests = np.empty_like(nests)
        for i in range(nest_count):
            new_nests[i] = nests[i] + levy_flight(dimension)
            new_nests[i] = np.clip(new_nests[i], lb, ub)

        # Evaluate new solutions
        new_fitness = np.apply_along_axis(objective_function, 1, new_nests)

        # Choose better solutions
        better_nests = new_fitness < fitness
        nests[better_nests] = new_nests[better_nests]
        fitness[better_nests] = new_fitness[better_nests]

        # Find the current best
        current_fmin, current_best = min(zip(fitness, nests))
        if current_fmin < fmin:
            fmin = current_fmin
            best_nest = current_best

        # Abandon some nests
        abandoned = np.random.rand(nest_count) < pa
        nests[abandoned] = np.random.uniform(lb, ub, (abandoned.sum(), dimension))

    return fmin, best_nest


def levy_flight(dimension):
    sigma1 = (np.math.gamma(1 + 1.5) * np.sin(np.pi * 1.5 / 2) / (
                np.math.gamma((1 + 1.5) / 2) * 1.5 * 2 ** ((1.5 - 1) / 2))) ** (1 / 1.5)
    sigma2 = 1
    u = np.random.normal(0, sigma1, dimension)
    v = np.random.normal(0, sigma2, dimension)
    step = u / abs(v) ** (1 / 1.5)
    return 0.01 * step