import numpy as np
import scipy
from scipy.stats import lognorm
from scipy.optimize import curve_fit


def lognorm_model(i, auc, mean, std):
    return [(auc / (x * std * np.sqrt(2 * np.pi))) * np.exp(((np.log(x) - mean) ** 2) / (2 * std ** 2)) for x in i]


class LOGNORM:
    fit: list
    params_range: list

    def __init__(self, x, time, auc=None, mean=None, std=None):
        if auc is None and mean is None and std is None:
            std = np.std(x)
            mean = np.exp(np.mean(x) + 0.5 * std ** 2)
            parameters, covariance = curve_fit(lognorm_model, time, x, p0=[mean, std], maxfev=200000)
            self.fit = lognorm_model(time, parameters[0], parameters[1])
            self.params_range = parameters
        elif auc is not None and mean is not None and std is not None:
            self.fit = lognorm_model(time, auc, mean, std)
