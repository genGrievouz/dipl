import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def ldrw_model(signal, auc, lamd, mu):
    return [auc * (np.exp(lamd) / lamd) * np.sqrt((mu * lamd) / (x * 2 * np.pi)) * np.exp(
        -0.5 * lamd * ((mu / x) + (x / mu))) for x in signal]


class LDRW:
    fit: list

    def __init__(self, x, time):
        parameters, covariance = curve_fit(ldrw_model, time, x, p0=[1, 1, 1], maxfev=200000)
        self.fit = ldrw_model(time, parameters[0], parameters[1], parameters[2])
