import numpy as np

from dipl.modules.ModAlgoAllModels import ModAlgoAllModels
from dipl.modules.ModuleModelData1 import ModuleModelData1
from dipl.modules.ModuleModelData2 import ModuleModelData2
from dipl.data.visualize import plot_result_algos
from dipl.evolutionary.ant_colony import ant_colony_optimization
from dipl.evolutionary.atrificial_bee_colony import abc
from dipl.evolutionary.cuckoo_search_v2 import cuckoo_search
from dipl.data.load_data import dataset_2_load_file_n
from dipl.evolutionary.firefly_algorithm import firefly_algorithm
from dipl.functions.gamma import gamma_model


# from evolutionary.cuckoo_search_v3 import run

def gamma_func(params, signal):
    auc, alpha, beta = params
    output = gamma_model(auc, alpha, beta, signal)
    return np.sum(np.abs(output))

def run_all():

    name = "per02_2_4_trig_DR60_inp_con_mreg_121113.mat"

    signal, time, ts = dataset_2_load_file_n(name)
    # signal, time, ts = dataset_1_load_file_n(name)

    #AUC, BETA, ALPHA
    # param_ranges = [(21000, 23000), (40, 50), (0.7, 1)]
    # param_ranges = [(0.1, 1000), (0.1, 100), (0.1, 10)]

    ModAlgoAllModels(name, signal, time, "abc")
    # get_params_cuckoo_v2()
    # get_params_for_cuckoo()
    # run_function()
    # run()
    # p_ant = ant_colony_optimization(signal, time, func)
    # p_pso, c = pso(signal, time, func)
    # p_cuc = cuckoo_search(signal, time, func)[1]
    # p_abc = abc(signal, time, param_ranges)
    # p_ant = ant_colony_optimization(signal, time, param_ranges)
    # p_fir = firefly_algorithm(signal, time, ts, param_ranges)[0]
    # plot_result_algos("gamma variate", signal, time, "gamma", p_cuc, p_abc, p_pso, p_ant, p_fir)


if __name__ == '__main__':
    run_all()
