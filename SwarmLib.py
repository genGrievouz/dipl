import os
from swarmlib import CuckooProblem, FUNCTIONS
from pymatreader import pymatreader
import numpy as np
import math
from pyswarm import pso

from dipl.CuckooModule3 import load_file_n
from dipl.functions.gamma import gamma_model


def load_file_n(name):
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


def model_function(params, signal):
    auc, beta, alpha = params
    y = gamma_model(signal, auc, beta, alpha)
    return y

param_ranges = [(0.1, 1.0), (0.1, 10.0), (0.1, 100.0)]

name = "per02_2_4_trig_DR60_inp_con_mreg_121113.mat"

signal, time, t = load_file_n(name)

best_params = pso(model_function, lb=[p[0] for p in param_ranges], ub=[p[1] for p in param_ranges], args=[signal], maxiter=100, swarmsize=20)

print("Best parameters:", best_params)