import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from scipy.optimize import curve_fit
from scipy.stats import gamma


def gamma_model(signal, auc, beta, alpha):
    print(type(signal))
    return [(auc * (x ** alpha)  * np.exp(-1 * x / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for x in signal]

def gamma_model_2(signal, k, sigma):
    return [(1 / (math.gamma(k) * (sigma ** k))) * (x ** (k-1)) * np.exp(-1 * x / sigma) for x in signal]

def gamma_model_3(signal, auc, k, sigma):
    return auc

class GAMMA:
    fit: list
    
    def  __init__(self, x, time):
        parameters, covariance = curve_fit(gamma_model, time, x, p0=[1,1,1], maxfev=200000)
        self.fit = gamma_model(time, parameters[0], parameters[1], parameters[2])
        # param = scipy.stats.gamma.fit(x)
        # # #x = normalize(x)
        # # xlin = np.linspace(0,np.max(x),len(x))
        # self.fit = scipy.stats.gamma.pdf(x, param[0], loc=param[1], scale=param[2])
        