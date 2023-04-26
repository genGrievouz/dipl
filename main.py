import pyswarms
import numpy as np

from dipl.ModuleModelData1 import ModuleModelData1
from dipl.ModuleModelData2 import ModuleModelData2
from dipl.ModulePso import get_params_pso
from dipl.data.visualize import plot_result, plot_result_algos
from dipl.evolutionary.ant_colony import ant_colony_optimization
from dipl.evolutionary.atrificial_bee_colony import abc
# from dipl.evolutionary.cuckoo_search import cuckoo_search
from dipl.evolutionary.cuckoo_search_v2 import cuckoo_search
from dipl.data.load_data import dataset_2_load_file_n
from dipl.CuckooModule import get_params_cuckoo_search


# from evolutionary.cuckoo_search_v3 import run

def run_all():

    name = "per02_2_4_trig_DR60_inp_con_mreg_121113.mat"

    signal, time, ts = dataset_2_load_file_n(name)
    #AUC, BETA, ALPHA
    param_ranges = [(21000, 23000), (40, 50), (0.7, 1)]
    # ModuleModelData2()
    # ModuleModelData1()
    # get_params_cuckoo_v2()
    # get_params_for_cuckoo()
    # run_function()
    # run()
    # p_ant = ant_colony_optimization(param_ranges, signal, time, ts)
    p_pso, c = get_params_pso(param_ranges, signal, time, ts)
    # p_cuc = cuckoo_search(signal, time, ts, param_ranges)
    p_abc = abc(signal, time, ts)
    p_ant = ant_colony_optimization(signal, time, ts, param_ranges)
    plot_result_algos("gamma variate", signal, time, "gamma", None, p_abc, p_pso, None)


if __name__ == '__main__':
    run_all()
