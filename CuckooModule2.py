import numpy as np
import os

from matplotlib import pyplot as plt
from pymatreader import pymatreader

from dipl.functions.fpt import FPT, fpt_model
from dipl.functions.gamma import gamma_model, GAMMA
from dipl.functions.lagnorm import LAGG, laggednormal_model
from dipl.functions.ldrw import LDRW, ldrw_model
from dipl.functions.lognorm import LOGNORM, lognorm_model
from evolutionary.cuckoo_search import cuckoo_search


def get_params_cuckoo_v2():
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

    def objective_function_gamma(params):
        auc, alpha, beta = params
        y = gamma_model(signal, auc, beta, alpha)
        error = np.sum((np.array(y) - np.array(signal)) ** 2)
        return error

    def objective_function_lognorm(params):
        mean, std = params
        y = lognorm_model(signal, mean, std)
        error = np.sum((np.array(y) - np.array(signal)) ** 2)
        return error

    def objective_function_ldrw(params):
        auc, lamba, mu = params
        y = ldrw_model(signal, auc, lamba, mu)
        error = np.sum((np.array(y) - np.array(signal)) ** 2)
        return error

    def objective_function_fpt(params):
        auc, mu, lam = params
        y = fpt_model(signal, auc, mu, lam)
        error = np.sum((np.array(y) - np.array(signal)) ** 2)
        return error

    def objective_function_laggednormal(params):
        mu, sigm, lam = params
        y = laggednormal_model(signal, mu, sigm, lam)
        error = np.sum((np.array(y) - np.array(signal)) ** 2)
        return error

    name = 'per02_2_4_trig_DR60_inp_con_mreg_121113.mat'
    signal, time, ts = load_file_n(name)
    lb = [1e-5, 1e-5, 1e-5]
    ub = [10, 10, 10]

    params_gamma = cuckoo_search(objective_function=objective_function_gamma,
                                 dimension=3,
                                 lb=lb,
                                 ub=ub)

    print(params_gamma)
    print(f'Best parameters found: auc={params_gamma[0]}, '
          f'auc={params_gamma[1][0]}, '
          f'alpha={params_gamma[1][1]},'
          f'beta={params_gamma[1][2]}'
          )

    auc_gamma = params_gamma[1][0]
    alpha_gamma = params_gamma[1][1]
    beta_gamma = params_gamma[1][2]

    # LDRW
    params_ldrw = cuckoo_search(objective_function=objective_function_ldrw,
                                dimension=3,
                                lb=lb,
                                ub=ub)

    ldrw = LDRW(signal, time)
    fit_ldrw = ldrw.fit

    print(f'Best parameters found: auc={params_ldrw[0]}, '
          f'auc={params_ldrw[1][0]}, '
          f'alpha={params_ldrw[1][1]},'
          f'beta={params_ldrw[1][2]}'
          )

    auc_ldrw = params_gamma[1][0]
    lamd_ldrw = params_gamma[1][1]
    mu_ldrw = params_gamma[1][2]

    # FPT
    params_fpt = cuckoo_search(objective_function=objective_function_fpt,
                               dimension=3,
                               lb=lb,
                               ub=ub)

    fpt = FPT(signal, time)
    fit_fpt = fpt.fit

    print(f'Best parameters found: auc={params_fpt[0]}, '
          f'auc={params_fpt[1][0]}, '
          f'alpha={params_fpt[1][1]},'
          f'beta={params_fpt[1][2]}'
          )

    auc_fpt = params_gamma[1][0]
    lam_fpt = params_gamma[1][1]
    mu_fpt = params_gamma[1][2]

    # MODELS ALGORITHM

    # fit_lognormal_algo = lognorm_model(signal, mean, std)
    fit_gamma_algo = gamma_model(signal, auc_gamma, beta_gamma, alpha_gamma)
    fit_ldrw_algo = ldrw_model(signal, auc_ldrw, lamd_ldrw, mu_ldrw)
    fit_fpt_algo = fpt_model(signal, auc_fpt, mu_fpt, lam_fpt)
    # fit_lagged_algo = laggednormal_model(signal, auc, mu, sigm, lam)

    # MODELS

    gamma = GAMMA(signal, time)
    fit_gamma = gamma.fit

    fpt = FPT(signal, time)
    fit_fpt = fpt.fit

    lognorm = LOGNORM(signal, time, ts)
    fit_lognormal = lognorm.fit

    lagged = LAGG(signal, time)
    fit_lagged = lagged.fit

    plt.figure()
    plt.plot(time, signal, 'o', label='data')
    # lognormal
    # plt.plot(time, fit_lognormal, '-', label='lognormal')
    # plt.plot(time, fit_lognormal_algo, '-', label='gamma lognormal algo')
    # gamma variate
    # plt.plot(time, fit_gamma, '-', label='gamma variate')
    # plt.plot(time, fit_gamma_algo, '-', label='gamma variate algo')
    # # ldrw
    # plt.plot(time, fit_ldrw, '-', label='ldrw')
    # plt.plot(time, fit_ldrw_algo, '-', label='gamma ldrw algo')
    # # ftp
    # plt.plot(time, fit_fpt, '-', label='fpt')
    # plt.plot(time, fit_fpt_algo, '-', label='gamma ftp algo')
    # # lagged normal
    # plt.plot(time, fit_lagged, '-', label='lagged normal')
    # plt.plot(time, fit_lagged_algo, '-', label='gamma lagged algo')

    plt.title(name)
    plt.legend(loc='upper right')
    plt.ylabel('Intensity')
    plt.xlabel('Time [s]')
    plt.show()
