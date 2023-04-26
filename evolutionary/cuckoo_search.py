import os

from math import gamma, sin, pi
import numpy as np
from matplotlib import pyplot as plt
from pymatreader import pymatreader
from scipy.optimize import curve_fit
from dipl.functions.gamma import gamma_model


def objective_function_gamma(params, signal, y):
    auc, alpha, beta = params
    y_pred = gamma_model(signal, auc, beta, alpha)
    return np.mean((y - y_pred) ** 2)


def cuckoo_search(objective_function, lb, ub, dimension, n, max_iter):
    """
    Cuckoo Search Algorithm
    :param objective_function: The function to optimize
    :param lb: Lower bound of the search space
    :param ub: Upper bound of the search space
    :param dimension: The number of dimensions in the search space
    :param n: The number of nests (population size)
    :param max_iter: The maximum number of iterations
    :return: The best solution found
    """
    # Initialize nests randomly
    nests = np.random.uniform(lb, ub, (n, dimension))

    # Get the fitness of each nest
    fitness = np.apply_along_axis(objective_function, 1, nests)

    # Find the best nest
    fmin, best_nest = min(zip(fitness, nests))

    # Initialize parameters
    pa = 0.25  # Discovery rate of alien eggs/solutions
    alpha = 1.0  # Step size scaling factor

    # Start iterations
    for _ in range(max_iter):
        # Generate new solutions (but keep the current best)
        new_nests = np.empty_like(nests)
        for i in range(n):
            step_size = alpha * levy(dimension)
            new_nests[i] = nests[i] + step_size * (nests[i] - best_nest)

        # Apply bounds to new solutions
        np.clip(new_nests, lb, ub, out=new_nests)

        # Evaluate new solutions and update nests
        new_fitness = np.apply_along_axis(objective_function, 1, new_nests)
        nests[fitness > new_fitness] = new_nests[fitness > new_fitness]
        fitness[fitness > new_fitness] = new_fitness[fitness > new_fitness]

        # Find the current best nest and its fitness
        current_fmin, current_best_nest = min(zip(fitness, nests))

        # Update the overall best nest if necessary
        if fmin > current_fmin:
            fmin = current_fmin
            best_nest = current_best_nest

        # Abandon some nests and build new ones
        abandoned_nests = np.random.rand(n) < pa
        nests[abandoned_nests] = np.random.uniform(lb, ub, (abandoned_nests.sum(), dimension))

    return best_nest


def levy(dimension):
    """
    Generate step sizes for Cuckoo Search using Levy flights.
    :param dimension: The number of dimensions in the search space.
    :return: A vector of step sizes.
    """
    beta = 3 / 2
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (
            1 / beta)

    u = np.random.normal(0, sigma, dimension)
    v = np.random.normal(0, 1, dimension)

    return u / abs(v) ** (1 / beta)