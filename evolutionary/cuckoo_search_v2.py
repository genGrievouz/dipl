import numpy as np
import math


def cuckoo_search(signal: list,
                  time: list,
                  ts: float,
                  param_ranges: list
                  ):
    def objective_function(params):
        auc, alpha, beta = params
        output = [(auc * (x ** alpha) * np.exp(-1 * x / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for x in
                  signal]
        return np.sum(np.abs(output))

    n_nests = 100
    n_iterations = 1000
    pa = 0.25
    dimension = 3
    lower_bound = [p[0] for p in param_ranges]
    upper_bound = [p[1] for p in param_ranges]

    # Initialize nests
    nests = np.random.uniform(lower_bound, upper_bound, (n_nests, dimension))

    # Evaluate nests
    nest_fitness = np.apply_along_axis(objective_function, 1, nests)

    # Find the current best nest
    fmin, best_nest = min(nest_fitness), nests[nest_fitness.argmin()]

    # Start iterations
    for _ in range(n_iterations):
        # Get a cuckoo and generate a new solution
        cuckoo = nests[np.random.randint(n_nests)]
        new_solution = cuckoo + np.random.normal(0, 1, dimension) * (best_nest - cuckoo)

        # Evaluate the new solution
        new_solution_fitness = objective_function(new_solution)

        # Choose a nest at random and compare its fitness to that of the new solution
        j = np.random.randint(n_nests)
        if new_solution_fitness < nest_fitness[j]:
            nests[j] = new_solution
            nest_fitness[j] = new_solution_fitness

        # Abandon a fraction of the worst nests and build new ones
        n_abandoned = int(n_nests * pa)
        worst_nests = np.argsort(nest_fitness)[-n_abandoned:]
        nests[worst_nests] = np.random.uniform(lower_bound, upper_bound, (n_abandoned, dimension))

        # Evaluate the new nests
        nest_fitness[worst_nests] = np.apply_along_axis(objective_function, 1, nests[worst_nests])

        # Find the current best nest
        current_fmin, current_best_nest = min(nest_fitness), nests[nest_fitness.argmin()]

        # Update the global best nest if necessary
        if current_fmin < fmin:
            fmin, best_nest = current_fmin, current_best_nest

    print(fmin, best_nest)

    return fmin, best_nest
