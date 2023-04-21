import os
import scipy
import numpy as np
import pymatreader

from scipy import signal


def load_file(name) -> dict:
    data_path = os.getcwd() + '\data\dataset_1'
    data_f = {}
    data = pymatreader.read_mat(data_path + "\\" + name)
    signal = data['tissue'][0]
    T_s = data['info']['acq']["Ts"]
    return {'signal': data['tissue'][0],
            'time': np.linspace(0, 0 + (T_s * len(signal)), len(signal), endpoint=False),
            'ts': T_s}


def load_data() -> dict:
    data_path = os.getcwd() + '\data\dataset_1'
    data = {}
    for file in os.listdir(data_path):
        if ".mat" in file:
            data[file.replace('.mat', '')] = load_file(file)
    return data


def preprocessing(x) -> list:

    def smooth_savgol(x) -> list:
        return signal.savgol_filter(x, 7, 3)

    def get_condition(x):
        x_mean = np.mean(x)
        x_min = min(x)
        return np.mean([x_min, x_min])

    def find_first_seq(x):
        pos = 0
        condition = get_condition(x)
        for i in range(len(x) - 1):
            if abs(x[i] - x[i + 1]) >= np.mean(x[:i]):
                break
            else:
                pos += 1
        return np.mean(x[:pos]), pos

    def preprocess(x):
        dev, c = find_first_seq(x)
        for i in range(len(x)):
            x[i] -= dev
        return x

    def remove_negative_values(x):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0.1
        return x

    # return smooth_savgol(remove_negative_values(preprocess(x)))
    return remove_negative_values(preprocess(x))


def resample(x, txt) -> list:
    def ma(x):
        max_ = np.argmax(x)
        min_ = np.argmin(x)
        arr = np.where(x == min_)
        # min_ = arr[0][len(arr[0])-1]
        x = x.tolist()
        max_ind = x.index(np.max(x))
        min_ind = x.index(np.min(x))
        return x[min_:max_], max_ind, min_ind

    s, max, min = ma(x)

    if txt == 'min':
        return scipy.signal.resample(x[:min], len(x[:min]) * 2)
    if txt == 'mid':
        return scipy.signal.resample(x[min:max], len(x[min:max]) * 2)
    if txt == 'max':
        return scipy.signal.resample(x[max:], len(x[max:]) * 2)


def normalize(x) -> list:
    new_x = []
    for i in x:
        new_x.append((i - np.min(x)) / (np.max(x) - np.min(x)))
    return new_x


def smooth_savgol(x) -> list:
    return signal.savgol_filter(x, 10, 3)


def remove_negative_values(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.1
    return x


def find_rising_point(signal):
    differences = np.diff(signal)
    index = np.argmax(differences > 0)
    rising_point = signal[index]
    print(f'Signal starts rising at index {index}, value {rising_point}')
