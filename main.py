import os

import numpy as np

from dipl.modules.ModuleDoAll import ModuleDoAll
from dipl.modules.ModAlgoAllModels import ModAlgoAllModels
from dipl.modules.ModuleModelData1 import ModuleModelData1
from dipl.modules.ModuleModelData2 import ModuleModelData2
from dipl.data.load_data import dataset_2_load_file_n, dataset_1_load_file_n
from dipl.evolutionary.firefly_algorithm import firefly_algorithm
from dipl.functions.gamma import gamma_model

def run_all():

    """
    For visualizing a datasets
    ModuleModelData1() - dataset 1
    ModuleModelData2() - dataset 2
    """
    # ModuleModelData1()
    # ModuleModelData2()


    """
    Example of loaing and displaying all models for
    a one signal from dataset 2
    You can chose between library algorithms:
    "pso", "de"
    or:
    "ant", "cuckoo search", 
    "fir", "abc", "spider monkey"  
    """
    # name = "per02_2_4_trig_DR60_inp_con_mreg_121113.mat"
    # signal, time, ts = dataset_2_load_file_n(name)
    # ModAlgoAllModels(name, signal, time, "de")


    """
    Example of loading and displaying all models for dataset 1
    """
    # name = "exp13_aifnova_inp_tis_111017.mat"
    # signal, time, ts = dataset_1_load_file_n(name)
    # ModAlgoAllModels(name, signal, time, "ant")


    """
    Example of loading and displaying all models for dataset 1 or 2
    At the end it will prints r2, spearman and nrmse for all files
    """
    r_2 = []
    spearman = []
    nrmse = []

    signals = os.listdir("data/dataset_1")

    for file_name in signals:
        for alg in ["de"]:
            signal, time, ts = dataset_1_load_file_n(file_name)
            m = ModAlgoAllModels(file_name, signal, time, alg)
            r_2.append(m.r_2)
            spearman.append(m.spearman)
            nrmse.append(m.nrmse)

    def get_r2():
        r2_lognormal = []
        r2_gamma = []
        r2_ldrw = []
        r2_fpt = []
        r2_lagged = []
        for i in r_2:
            r2_lognormal.append(i["lognormal"])
            r2_gamma.append(i["gamma"])
            r2_ldrw.append(i["ldrw"])
            r2_fpt.append(i["fpt"])
            r2_lagged.append(i["lagged"])
        print("r2_lognormal", np.mean(r2_lognormal))
        print("r2_gamma", np.mean(r2_gamma))
        print("r2_ldrw", np.mean(r2_ldrw))
        print("r2_fpt", np.mean(r2_fpt))
        print("r2_lagged", np.mean(r2_lagged))

    def get_spearman():
        spearman_lognormal = []
        spearman_gamma = []
        spearman_ldrw = []
        spearman_fpt = []
        spearman_lagged = []
        for i in spearman:
            spearman_lognormal.append(i["lognormal"])
            spearman_gamma.append(i["gamma"])
            spearman_ldrw.append(i["ldrw"])
            spearman_lagged.append(i["lagged"])
            spearman_fpt.append(i["fpt"])
        print("spearman_lognormal", np.mean(spearman_lognormal))
        print("spearman_gamma", np.mean(spearman_gamma))
        print("spearman_ldrw", np.mean(spearman_ldrw))
        print("spearman_fpt", np.mean(spearman_fpt))
        print("spearman_lagged", np.mean(spearman_lagged))

    def ger_nrmse():
        nrmse_gamma = []
        nrmse_ldrw = []
        nrmse_fpt = []
        nrmse_lagged = []
        for i in nrmse:
            nrmse_lagged.append(i["lognormal"])
            nrmse_gamma.append(i["gamma"])
            nrmse_ldrw.append(i["ldrw"])
            nrmse_fpt.append(i["fpt"])
            nrmse_lagged.append(i["lagged"])
        print("nrmse_lognormal", np.mean(nrmse_lagged))
        print("nrmse_gamma", np.mean(nrmse_gamma))
        print("nrmse_ldrw", np.mean(nrmse_ldrw))
        print("nrmse_fpt", np.mean(nrmse_fpt))
        print("nrmse_lagged", np.mean(nrmse_lagged))


    get_r2()
    get_spearman()
    ger_nrmse()


if __name__ == '__main__':
    run_all()