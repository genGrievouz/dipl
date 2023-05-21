import pygad

from dipl.data.search_space import get_objective_function_and_params


def ga_algorithm(
        signal: list,
        time: list,
        objective_function_type: str
):
    param_ranges, objective_function = get_objective_function_and_params(signal, time, objective_function_type)
    param_list = [list(tup) for tup in param_ranges]

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=5,
        fitness_func=objective_function,
        sol_per_pop=10,
        num_genes=3,
        gene_space=param_list,
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print(solution)
    return solution
