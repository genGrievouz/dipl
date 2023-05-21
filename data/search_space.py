from dipl.functions.fpt import fpt_model, FPT
from dipl.functions.gamma import GAMMA, gamma_model
from dipl.functions.lagnorm import laggednormal_model, LAGG
from dipl.functions.ldrw import ldrw_model, LDRW
from dipl.functions.lognorm import lognorm_model, LOGNORM

import numpy as np


def objective_function_gamma(params, signal):
    auc, beta, alpha = params
    output = gamma_model(signal, auc, beta, alpha)
    return np.sum(np.abs(output))


def objective_function_fpt(params, signal):
    auc, lam, mu = params
    output = fpt_model(signal, auc, lam, mu)
    return np.sum(np.abs(output))


def objective_function_lognormal(params, signal):
    auc, mean, std = params
    output = lognorm_model(signal, auc, mean, std)
    return np.sum(np.abs(output))


def objective_function_lagged(params, signal):
    auc, lam, mu, sigm = params
    output = laggednormal_model(signal, auc, lam, mu, sigm)
    return np.sum(np.abs(output))


def objective_function_ldrw(params, signal):
    auc, lamd, mu = params
    output = ldrw_model(signal, auc, lamd, mu)
    return np.sum(np.abs(output))


def to_model(signal, time, model, params):
    if model == "gamma":
        gamma = GAMMA(signal, time, params[0], params[1], params[2])
        print(gamma.fit)
        return gamma.fit
    if model == "fpt":
        ftp = FPT(signal, time, params[0], params[1], params[2])
        return ftp.fit
    if model == "lagged":
        lagg = LAGG(signal, time, params[0], params[1], params[2], params[3])
        return lagg.fit
    if model == "lognormal":
        lognormal = LOGNORM(signal, time, params[0], params[1], params[2])
        return lognormal.fit
    if model == "ldrw":
        ldrw = LDRW(signal, time, params[0], params[1], params[2])
        return ldrw.fit


def get_objective_function_and_params(signal, time, objective_function_type):
    global objective_function, param_ranges

    if objective_function_type == "gamma":
        param_ranges = define_search_space(GAMMA(signal, time).params_range)
        objective_function = objective_function_gamma
    if objective_function_type == "fpt":
        param_ranges = define_search_space(FPT(signal, time).params_range)
        objective_function = objective_function_fpt
    if objective_function_type == "lagged":
        param_ranges = define_search_space(LAGG(signal, time).params_range)
        objective_function = objective_function_lagged
    if objective_function_type == "lognormal":
        param_ranges = define_search_space(LOGNORM(signal, time).params_range)
        objective_function = objective_function_lognormal
    if objective_function_type == "ldrw":
        param_ranges = define_search_space(LDRW(signal, time).params_range)
        objective_function = objective_function_ldrw

    return param_ranges, objective_function


def define_search_space(arr):
    result = []
    for x in arr:
        lower_bound, upper_bound = create_boundary(x)
        result.append((lower_bound, upper_bound))
    return result


def create_boundary(number):
    if isinstance(number, int):
        lower = number - 1
        upper = number + 1
    else:
        offset = 0.1 * abs(number)
        lower = number - offset
        upper = number + offset
    return (lower, upper)
