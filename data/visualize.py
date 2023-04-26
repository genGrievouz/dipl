import matplotlib.pyplot as plt

from dipl.functions.fpt import FPT
from dipl.functions.gamma import GAMMA
from dipl.functions.lagnorm import LAGG
from dipl.functions.ldrw import LDRW
from dipl.functions.lognorm import LOGNORM


def plot_result(fig_name, x, time, ts, params, algo: str):
    # ldrw = LDRW(x, time)
    # fit_ldrw = ldrw.fit

    gamma = GAMMA(x, time)
    fit_gamma = gamma.fit

    gama_algo = GAMMA(x, time, auc=params[0], beta=params[1], alpha=params[2])
    fit_gamma_algo = gama_algo.fit

    # fpt = FPT(x, time)
    # fit_fpt = fpt.fit
    #
    # lognorm = LOGNORM(x, time, ts)
    # fit_lognormal = lognorm.fit
    #
    # lagged = LAGG(x, time)
    # fit_lagged = lagged.fit

    plt.figure()
    plt.plot(time, x, 'o', label='data')
    # plt.plot(time, fit_lognormal, '-', label='lognormal')
    plt.plot(time, fit_gamma, '-', label='gamma variate')
    plt.plot(time, fit_gamma_algo, '-', label='gamma variate ' + algo)
    # plt.plot(time, fit_ldrw, '-', label='ldrw')
    # plt.plot(time, fit_fpt, '-', label='fpt')
    # plt.plot(time, fit_lagged, '-', label='lagged normal')
    # plt.title(name)
    plt.legend(loc='upper right')
    plt.ylabel('Intensity')
    plt.xlabel('Time [s]')
    plt.show()


def plot_result_algos(
        fig_name: str,
        x: list,
        time: list,
        model: str,
        params_cuckoo: list or None,
        params_abc: list or None,
        params_pso: list or None,
        params_ant: list or None,
):
    global fit_gamma_algo_cuc
    global fit_gamma_algo_abc
    global fit_gamma_algo_pso
    global fit_gamma_algo_ant

    if model == "gamma":
        gamma = GAMMA(x,
                      time
                      )
        fit_gamma = gamma.fit

        if params_cuckoo is not None:
            gamma_algo_cuc = GAMMA(x,
                                   time,
                                   auc=params_cuckoo[0],
                                   beta=params_cuckoo[1],
                                   alpha=params_cuckoo[2]
                                   )
            fit_gamma_algo_cuc = gamma_algo_cuc.fit
            print("calculated cuckoo")

        if params_abc is not None:
            gamma_algo_abc = GAMMA(x,
                                   time,
                                   auc=params_abc[0],
                                   beta=params_abc[1],
                                   alpha=params_abc[2]
                                   )
            fit_gamma_algo_abc = gamma_algo_abc.fit
            print("calc abc")

        if params_pso is not None:
            gamma_algo_pso = GAMMA(x,
                                   time,
                                   auc=params_pso[0],
                                   beta=params_pso[1],
                                   alpha=params_pso[2]
                                   )
            fit_gamma_algo_pso = gamma_algo_pso.fit
            print("calc pso")

        if params_ant is not None:
            gamma_algo_ant = GAMMA(x,
                                   time,
                                   auc=params_ant[0],
                                   beta=params_ant[1],
                                   alpha=params_ant[2]
                                   )
            fit_gamma_algo_ant = gamma_algo_ant.fit
            print("calc ant")

        plt.figure()
        plt.plot(time, x, 'o', label='data')
        plt.plot(time, fit_gamma, '-', label=fig_name)
        if params_cuckoo is not None:
            plt.plot(time, fit_gamma_algo_cuc, '-', label=fig_name + " cuckoo")
        if params_abc is not None:
            plt.plot(time, fit_gamma_algo_abc, '-', label=fig_name + " abc")
        if params_pso is not None:
            plt.plot(time, fit_gamma_algo_pso, '-', label=fig_name + " pso")
        if params_ant is not None:
            plt.plot(time, fit_gamma_algo_ant, '-', label=fig_name + " ant")
        plt.legend(loc='upper right')
        plt.ylabel('Intensity')
        plt.xlabel('Time [s]')
        plt.show()
