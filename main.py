import numpy as np
import matplotlib.pyplot as plt

from functions.lognorm import LOGNORM
from functions.gamma import GAMMA
from functions.ldrw import LDRW
from functions.fpt import FPT
from functions.laggednormal import LAGG

from functions.load_data import load_data, preprocessing, normalize, remove_negative_values

from errors.errors import rmse, r2

from functions.load_data import load_data


def run_all():
    data = load_data()

    for k, v in data.items():
        x = normalize(remove_negative_values(preprocessing((data[k]['signal']))))
        ind = int(np.round(0.5 * len(x)))
        x = x[:ind]
        time = data[k]['time']
        time = time[:ind]
        ts = data[k]['ts']

        ldrw = LDRW(x, time)
        fit_ldrw = ldrw.fit

        gamma = GAMMA(x, time)
        fit_gamma = gamma.fit

        fpt = FPT(x, time)
        fit_fpt = fpt.fit

        lognorm = LOGNORM(x, time, ts)
        fit_lognormal = lognorm.fit

        lagged = LAGG(x, time)
        fit_lagged = lagged.fit

        plt.figure()
        plt.plot(time, x, 'o', label='data')
        plt.plot(time, fit_lognormal, '-', label='lognormal')
        plt.plot(time, fit_gamma, '-', label='gamma variate')
        plt.plot(time, fit_ldrw, '-', label='ldrw')
        plt.plot(time, fit_fpt, '-', label='fpt')
        plt.plot(time, fit_lagged, '-', label='lagged normal')
        plt.title(k)
        plt.legend(loc='upper right')
        plt.ylabel('Intensity')
        plt.xlabel('Time [s]')
        plt.show()


run_all()
