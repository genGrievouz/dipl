from swarmlib import ABCProblem

from dipl.evolutionary.atrificial_bee_colony import abc


def get_abc_params(signal, time, ts):
    return abc(signal, time, ts)