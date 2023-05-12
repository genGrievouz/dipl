import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def calculate_r2(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2 = r2_score(y_true, y_pred)
    return r2


def calculate_spearman(x, y):
    x = np.array(x)
    y = np.array(y)
    corr, pval = spearmanr(x, y)
    return corr


def calculate_nrmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / np.mean(y_true)
    return nrmse
