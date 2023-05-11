from pyswarm import pso
from dipl.data.search_space import get_objective_function_and_params


def pso_algorithm(signal: list,
                  time: list,
                  objective_function_type: str
                  ):

    param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)

    best_params = pso(
        objective_function,
        lb=[p[0] for p in param_ranges],
        ub=[p[1] for p in param_ranges],
        args=tuple([signal]),
        maxiter=100,
        swarmsize=10,
    )

    return best_params[0]
