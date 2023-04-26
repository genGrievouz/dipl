import math
import numpy as np
from pyswarm import pso

# define the objective function
def objective_function(params, signal):
    auc, alpha, beta = params
    output = [(auc * (x ** alpha) * np.exp(-1 * x / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for x in
              signal]
    return np.sum(np.abs(output))


def get_params_pso(
        param_ranges: list,
        signal: list,
        time: list,
        ts: float
):

    # define the signal data
    # signal = np.random.rand(100)

    # run the cuckoo search algorithm
    best_params = pso(
        objective_function,
        lb=[p[0] for p in param_ranges],
        ub=[p[1] for p in param_ranges],
        args=tuple([signal]),
        maxiter=100,
        swarmsize=10,
    )

    # print the best parameter values
    print("Best parameters:", best_params)

    return best_params
