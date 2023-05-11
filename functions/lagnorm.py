import numpy as np
import math

from scipy.optimize import curve_fit


def laggednormal_model(signal, auc, mu, sigm, lam):
    return [auc / 2 * lam * np.exp(
        -1 * lam * x - (mu ** 2) / (2 * sigm ** 2) + ((mu + lam * sigm ** 2) ** 2) / (2 * sigm ** 2)) * (
                        1 + math.erf((x - mu - lam * sigm ** 2) / np.sqrt(2 * sigm ** 2))) for x in signal]


class LAGG:
    fit: list
    params_range: list

    def __init__(self, x, time, auc=None, mu=None, sigm=None, lam=None):
        if auc is None and mu is None and sigm is None and lam is None:
            parameters, covariance = curve_fit(laggednormal_model, time, x, p0=[1, 1, 1, 1], maxfev=200000)
            self.fit = laggednormal_model(time, parameters[0], parameters[1], parameters[2], parameters[3])
            self.params_range = parameters
        elif auc is not None and mu is not None and sigm is not None and lam is not None:
            self.fit = laggednormal_model(time, auc, mu, sigm, lam)
