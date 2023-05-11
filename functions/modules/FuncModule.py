import numpy as np

from dipl.functions.fpt import fpt_model
from dipl.functions.gamma import gamma_model
from dipl.functions.lagnorm import laggednormal_model
from dipl.functions.ldrw import ldrw_model
from dipl.functions.lognorm import lognorm_model


def gamma_objective(params, signal):
    auc, alpha, beta = params
    output = gamma_model(signal, auc, beta, alpha)
    return np.sum(np.abs(output))


def fpt_objective(params, signal):
    auc, mu, lam = params
    output = fpt_model(signal, auc, mu, lam)
    return np.sum(np.abs(output))


def lagnorm_objective(params, signal):
    auc, mu, sigm, lam = params
    output = laggednormal_model(signal, auc, mu, sigm, lam)
    return np.sum(np.abs(output))


def ldrw_objective(params, signal):
    auc, lamd, mu = params
    output = ldrw_model(signal, auc, lamd, mu)
    return np.sum(np.abs(output))


def lognorm_objective(params, signal):
    auc, mean, std = params
    output = lognorm_model(signal, auc, mean, std)
    return np.sum(np.abs(output))


def objective_function_factory(objective_type, signal):
    if objective_type == "gamma":
        return lambda params: gamma_objective(params, signal)
    if objective_type == "fpt":
        return lambda params: fpt_objective(params, signal)
    if objective_type == "lagged":
        return lambda params: lagnorm_objective(params, signal)
    if objective_type == "ldrw":
        return lambda params: ldrw_objective(params, signal)
    if objective_type == "lognormal":
        return lambda params: lognorm_objective(params, signal)
    else:
        raise ValueError("Unsupported objective function type: {}".format(objective_type))
