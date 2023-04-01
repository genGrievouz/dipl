import sklearn.metrics
import math


def mse(x,p_x):
    return sklearn.metrics.mean_squared_error(x,p_x)

def rmse(x, p_x):
    return math.sqrt(mse(x,p_x))

def r2(x,p_x):
    return sklearn.metrics.r2_score(x,p_x)



