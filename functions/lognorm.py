import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import lognorm


def lognorm_model(i, auc, mean, std):
    return lognorm.pdf(i, std, loc=0, scale=np.exp(mean)) * auc

class LOGNORM:
    fit: list
    params_range: list

    def __init__(self, x, time, auc=None, mean=None, std=None):
        if auc is None and mean is None and std is None:
            parameters, covariance = curve_fit(lognorm_model, time, x, p0=[1, 1, 1], maxfev=200000)
            self.fit = lognorm_model(time, parameters[0], parameters[1], parameters[2])
            self.params_range = parameters
        elif auc is not None and mean is not None and std is not None:
            self.fit = lognorm_model(time, auc, mean, std)
