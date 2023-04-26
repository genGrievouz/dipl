import os

import numpy as np
from pymatreader import pymatreader


def dataset_2_load_file_n(name):
    data_path = os.getcwd() + '\data\dataset_2'
    data = pymatreader.read_mat(data_path + "\\" + name)
    signal = []
    dataset = data['data1']

    for i in range(0, len(dataset)):
        x = np.floor(dataset[i].shape[1] / 2)
        y = np.floor(dataset[i].shape[0] / 2)
        signal.append(dataset[i][int(y)][int(x)])

    x = signal
    time = data['info']["acq"]['TimeStamps']
    ts = data['info']['acq']["Ts"]
    return x, time, ts