import random
import math
import numpy as np

from dipl.data.search_space import get_objective_function_and_params
from dipl.functions.gamma import gamma_model

#TODO

def firefly_algorithm(signal,
                      time,
                      ts,
                      objective_function_type,
                      num_fireflies=10,
                      max_iterations=100,
                      alpha=0.5,
                      beta=0.5,
                      gamma=1.0):

    param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)

    num_parameters = len(param_ranges)
    lower_bounds = [p[0] for p in param_ranges]
    upper_bounds = [p[1] for p in param_ranges]

    """
    Implementation of the Firefly Algorithm for numerical optimization.

    Parameters:
    - objective_function (function): the function to be optimized
    - num_parameters (int): the number of parameters in the optimization problem
    - num_fireflies (int): the number of fireflies in the swarm (default: 50)
    - max_iterations (int): the maximum number of iterations (default: 100)
    - alpha (float): the attractiveness coefficient (default: 0.5)
    - beta (float): the absorption coefficient (default: 0.2)
    - gamma (float): the step size (default: 1.0)

    Returns:
    - best_solution (ndarray): the best solution found by the algorithm
    - best_fitness (float): the fitness value of the best solution found by the algorithm
    """

    # Initialize the fireflies
    fireflies = np.random.uniform(lower_bounds, upper_bounds, size=(num_fireflies, num_parameters))
    light_intensity = np.zeros(num_fireflies)

    # Evaluate the initial light intensity of each firefly
    for i in range(num_fireflies):
        light_intensity[i] = objective_function(fireflies[i])

    # Initialize the best firefly and its light intensity
    best_firefly = np.zeros(num_parameters)
    best_light_intensity = float('inf')

    # Main loop
    for t in range(max_iterations):

        # Update the attractiveness coefficient
        alpha_t = alpha * math.exp(-gamma * t)

        # Move each firefly towards brighter fireflies
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                if light_intensity[j] > light_intensity[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta_t = beta * math.exp(-gamma * r ** 2)
                    fireflies[i] += beta_t * (fireflies[j] - fireflies[i]) + alpha_t * (
                            np.random.rand(num_parameters) - 0.5)
                    # Enforce parameter bounds
                    fireflies[i] = np.maximum(fireflies[i], lower_bounds)
                    fireflies[i] = np.minimum(fireflies[i], upper_bounds)
                    light_intensity[i] = objective_function(fireflies[i])

        # Update the best firefly and its light intensity
        index = np.argmin(light_intensity)
        if light_intensity[index] < best_light_intensity:
            best_firefly = fireflies[index]
            best_light_intensity = light_intensity[index]

    return best_firefly, best_light_intensity
