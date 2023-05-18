from dipl.functions.lagnorm import LAGG
from dipl.functions.lognorm import LOGNORM
from dipl.functions.fpt import FPT
from dipl.functions.gamma import GAMMA
from dipl.functions.ldrw import LDRW

import pymatreader
import os
import numpy as np
import matplotlib.pyplot as plt


class ModuleModelData2:
    """
    Class for loading and preprocessing the data.
    Also displays the data.
    Directory: data/dataset_2
    """
    def load_file(self, name):
        data_path = os.getcwd() + '\data\dataset_2'
        data = pymatreader.read_mat(data_path + "\\" + name)
        signal = []
        dataset = data['data1']

        for i in range(0, len(dataset)):
            x = np.floor(dataset[i].shape[1] / 2)
            y = np.floor(dataset[i].shape[0] / 2)
            signal.append(dataset[i][int(y)][int(x)])

        x = signal
        threshold = min(x) + 0.1
        x = [i - threshold for i in x]
        time = data['info']["acq"]['TimeStamps']
        # ts = data['info']['acq']["Ts"]

        ldrw = LDRW(x, time)
        fit_ldrw = ldrw.fit

        gamma = GAMMA(x, time)
        fit_gamma = gamma.fit

        fpt = FPT(x, time)
        fit_fpt = fpt.fit

        lognorm = LOGNORM(x, time)
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
        plt.title(name)
        plt.legend(loc='upper right')
        plt.ylabel('Intensity')
        plt.xlabel('Time [s]')
        plt.show()

    def load_data(self):
        data_path = os.getcwd() + '\data\dataset_2'
        for file in os.listdir(data_path):
            if ".mat" in file:
                self.load_file(file)

    def __init__(self):
        self.load_data()