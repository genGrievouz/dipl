import os

import numpy as np
from pymatreader import pymatreader

from dipl.functions.load_data import preprocessing, remove_negative_values


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
    # min_x = min(x) + 1
    # x = [i - min_x for i in x]
    time = data['info']["acq"]['TimeStamps']
    ts = data['info']['acq']["Ts"]
    return x, time, ts


def dataset_1_load_file_n(name):
    data_path = os.getcwd() + '\data\dataset_1'
    data = pymatreader.read_mat(data_path + "\\" + name)
    x = data['tissue'][0]
    ts = data['info']['acq']["Ts"]
    time = np.linspace(0, 0 + (ts * len(x)), len(x), endpoint=False)
    x = remove_negative_values(preprocessing(x))
    threshold = min(x) + 0.1
    removed_indices = [index for index in range(len(x)) if x[index] < threshold]
    x = [i for i in x if i > threshold]
    time = [time[i] for i in range(len(time)) if i not in removed_indices]
    return x, time, ts
