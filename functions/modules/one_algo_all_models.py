from dipl.data.visualize import plot_result

from dipl.evolutionary.ant_colony import ant_colony_optimization
from dipl.evolutionary.pso import pso_algorithm
from dipl.evolutionary.atrificial_bee_colony import abc
from dipl.evolutionary.cuckoo_search import cuckoo_search
from dipl.evolutionary.firefly_algorithm import firefly_algorithm


def display_all_models(algorithm: str,
                       name: str,
                       signal: list,
                       time: list):

    global algo
    if algorithm == "pso":
        algo = pso_algorithm
    if algorithm == "cuc":
        algo = cuckoo_search
    if algorithm == "abc":
        algo = abc
    if algorithm == "ant":
        algo = ant_colony_optimization
    if algorithm == "fir":
        algo = firefly_algorithm

    gamma = algo(signal, time, "gamma")
    fpt = algo(signal, time, "fpt")
    lagged = algo(signal, time, "lagged")
    lognormal = algo(signal, time, "lognormal")
    ldrw = algo(signal, time, "ldrw")

    plot_result(
        name + " " + algorithm,
        signal,
        time,
        gamma, fpt, lagged, lognormal, ldrw
    )
