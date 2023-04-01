import numpy as np
import scipy
import paramnormal
import math
from scipy.stats import lognorm
from scipy.optimize import curve_fit

from functions.load_data import normalize, remove_negative_values


def model(i, auc, mean, std):
    return [(auc / (x * std * np.sqrt(2 * np.pi))) * np.exp(((np.log(x) - mean) ** 2) / (2 * std ** 2)) for x in i]


def lognorm_model(signal, mean, std):
    return [(1 / (x * std * np.sqrt(2 * np.pi))) * np.exp(-1 * (np.log(x) - mean) ** 2 / (2 * std ** 2)) for x in
            signal]


def lognorm_model_2(x, std):
    return 1 / (x * std * np.sqrt(2 * np.pi)) * np.exp(-1 * (np.log(x)) ** 2 / (2 * std ** 2))


def mean_m(x):
    return np.sum(np.log(x)) / len(x)


def std_m(x):
    return np.sum(np.log(x) - mean_m(x)) / len(x)


class LOGNORM:
    fit: list
    fit2: list
    fit3: list

    def __init__(self, x, time, t):
        std = np.std(x)
        mean = np.mean(x)
        mean = np.exp(np.mean(x) + 0.5 * std ** 2)
        # mean = mean_m(x)
        # var = np.exp((std ** 2)- 1) * np.exp(2 * np.mean(x) + (std ** 2))
        # std_2 =
        parameters, covariance = curve_fit(lognorm_model, time, x, p0=[mean, std], maxfev=200000)
        self.fit = lognorm_model(time, parameters[0], parameters[1])
        # parameters, covariance = curve_fit(model, time, x, p0=[1,mean,std], maxfev=200000)
        # self.fit = model(time, parameters[0], parameters[1], parameters[2])

        param = scipy.stats.lognorm.fit(np.exp(x), loc=std, scale=np.exp(mean))
        t = np.linspace(np.min(np.exp(x)), np.max(np.exp(x)), len(x))
        # lognorm_dist = scipy.stats.lognorm(s=std, loc=0, scale=np.mean(mean))
        self.fit2 = scipy.stats.lognorm.pdf(t, *param)

        # params = paramnormal.lognormal.fit(x)
        # dist = paramnormal.lognormal.from_params(params)
        # self.fit3 = dist.pdf
