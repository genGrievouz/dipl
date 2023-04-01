from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def fpt_model(signal, auc, mu, lam):
    return [auc * (np.exp(lam) / mu) * np.sqrt(lam / (2 * np.pi)) * ((mu / x) ** 1.5) * np.exp(
        -0.5 * lam * (mu / x + x / mu)) for x in signal]


class FPT:
    fit: list

    def __init__(self, x, time):
        parameters, covariance = curve_fit(fpt_model, time, x, p0=[1, 1, 1], maxfev=200000)
        self.fit = fpt_model(time, parameters[0], parameters[1], parameters[2])
