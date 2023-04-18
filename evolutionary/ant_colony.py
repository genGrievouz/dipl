# define the ant colony optimization algorithm
import numpy as np


def ant_colony_optimization(objective_func, lb, ub, n_ants=10, n_iterations=1000):
    # initialize the pheromone matrix
    pheromone = np.ones((n_ants, len(lb)))

    # initialize the best solution and its fitness
    best_solution = None
    best_fitness = np.inf

    # main loop
    for _ in range(n_iterations):
        # initialize the solutions and their fitnesses
        solutions = np.zeros((n_ants, len(lb)))
        fitnesses = np.zeros(n_ants)

        # for each ant
        for i in range(n_ants):
            # construct a solution
            for j in range(len(lb)):
                solutions[i, j] = lb[j] + (ub[j] - lb[j]) * np.random.rand()

            # evaluate the fitness of the solution
            fitnesses[i] = objective_func(solutions[i])

            # update the best solution and its fitness
            if fitnesses[i] < best_fitness:
                best_solution = solutions[i]
                best_fitness = fitnesses[i]

        # update the pheromone matrix
        pheromone *= 0.9
        pheromone[np.argmin(fitnesses)] += 0.1

    return best_solution
