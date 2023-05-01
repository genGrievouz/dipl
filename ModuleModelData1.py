from dipl.functions.lagnorm import LAGG
from dipl.functions.load_data import load_data, remove_negative_values, preprocessing, normalize
from dipl.functions.lognorm import LOGNORM
from dipl.functions.fpt import FPT
from dipl.functions.gamma import GAMMA
from dipl.functions.ldrw import LDRW
import matplotlib.pyplot as plt


class ModuleModelData1:
    """
    Class for loading and preprocessing the data.
    Directory: data/dataset_1
    Also displays the data.
    """
    def load_data(self):

        data = load_data()

        for k, v in data.items():

            x = normalize(remove_negative_values(preprocessing((data[k]['signal']))))
            threshold = 0.03
            removed_indices = [index for index in range(len(x)) if x[index] < threshold]
            x = [i for i in x if i > threshold]
            time = data[k]['time']
            time = [time[i] for i in range(len(time)) if i not in removed_indices]

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

    def __init__(self):
        self.load_data()