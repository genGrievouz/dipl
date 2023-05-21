from scipy.optimize import differential_evolution

from dipl.data.search_space import get_objective_function_and_params


def de_algorithm(
    signal: list,
    time: list,
    objective_function_type: str
    ):

    param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)

    best_params = differential_evolution(
        objective_function,
        bounds=param_ranges,
        args=tuple([signal]),
        maxiter=100,
    )

    return best_params.x
