import os

import numpy as np
import math

from dipl.evolutionary.cuckoo_search import cuckoo_search
from dipl.functions.gamma import gamma_model


def get_params_cuckoo_search(signal, time, ts, y):
    def objective_function(params):
        auc, alpha, beta = params
        y_pred = gamma_model(signal, auc, beta, alpha)
        return np.mean((y - y_pred) ** 2)

    params = cuckoo_search(objective_function, [0, 0, 0], [1, 1, 1], 3, 100, 1000)

    return params
