from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
import math

from functions.load_data import load_data, preprocessing, normalize, remove_negative_values
# from load_data import load_data, preprocessing, normalize, remove_negative_values
from scipy.special import gamma
from scipy.optimize import curve_fit


def laggednormal_model(signal, auc, mu, sigm, lam):
    return [auc / 2 * lam * np.exp(
        -1 * lam * x - (mu ** 2) / (2 * sigm ** 2) + ((mu + lam * sigm ** 2) ** 2) / (2 * sigm ** 2)) * (
                        1 + math.erf((x - mu - lam * sigm ** 2) / np.sqrt(2 * sigm ** 2))) for x in signal]


class LAGG:
    fit: list

    def __init__(self, x, time):
        parameters, covariance = curve_fit(laggednormal_model, time, x, p0=[1, 1, 1, 1], maxfev=200000)
        self.fit = laggednormal_model(time, parameters[0], parameters[1], parameters[2], parameters[3])
