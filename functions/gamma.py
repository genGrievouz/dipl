import numpy as np
import math
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.stats import gamma
#TODO save with zero np.devide


def gamma_model(signal, auc, beta, alpha):
    return [(auc * (x ** alpha) * np.exp(-1 * x / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for x in
            signal]


class GAMMA:
    fit: list
    params_range: list

    def __init__(self, x, time, auc=None, beta=None, alpha=None):
        if auc is None and beta is None and alpha is None:
            parameters, covariance = curve_fit(gamma_model, time, x, p0=[1, 1, 1], maxfev=200000)
            self.fit = gamma_model(time, parameters[0], parameters[1], parameters[2])
            self.params_range = parameters
        if auc is not None and beta is not None and alpha is not None:
            self.fit = gamma_model(time, auc, beta, alpha)
