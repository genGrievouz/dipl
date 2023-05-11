import sklearn.metrics
import math
from scipy.stats import spearmanr


def mse(x, p_x):
    """
    mean square error
    x - real values
    p_x - predicted values
    """
    return sklearn.metrics.mean_squared_error(x, p_x)


def rmse(x, p_x):
    """
    root mean square error
    x - real values
    p_x - predicted values
    """
    return math.sqrt(mse(x, p_x))


def r2(x, p_x):
    """
    r2 score
    x - real values
    p_x - predicted values
    """
    return sklearn.metrics.r2_score(x, p_x)


def nrmse(x, p_x):
    """
    normalized root mean square error
    x - real values
    p_x - predicted values
    """
    return rmse(x, p_x) / (max(x) - min(x))


def spearman_correlation(x, p_x):
    """
    spearman correlation
    x - real values
    p_x - predicted values
    """
    return spearmanr(x, p_x)[0]
