import os
import numpy as np

from dipl.conf import DIR
from dipl.data.load_data import dataset_2_load_file_n, dataset_1_load_file_n
from dipl.modules.AlgoWithAllModels import AlgoWithAllModels
from dipl.modules.ModAlgoAllModels import ModAlgoAllModels


class ModuleDoAll:
    file_names: list = []
    stats: dict = {}
    modules: list = []
    algorithm: str = ["cuckoo search", "pso",
                      "fir", "abc", "spider monkey"]

    def __init__(self, dataset: str):
        if dataset == "1":
            path = DIR + "\\data\\dataset_1"
            self.file_names = os.listdir(path)
            for name in self.file_names:
                signal, time, ts = dataset_1_load_file_n(name)
                for algorithm in self.algorithm:
                    self.modules.append(ModAlgoAllModels(name, signal, time, algorithm))

        if dataset == "2":
            path = DIR + "\\data\\dataset_2"
            self.file_names = os.listdir(path)
            for name in self.file_names:
                modul = AlgoWithAllModels()
                signal, time, ts = dataset_2_load_file_n(name)
                modul.name = name
                for algo in self.algorithm:
                    if algo == "cuckoo search":
                        modul.cuckoo_search = ModAlgoAllModels(name, signal, time, algo)
                    if algo == "pso":
                        modul.pso = ModAlgoAllModels(name, signal, time, algo)
                    if algo == "ant":
                        modul.ant = ModAlgoAllModels(name, signal, time, algo)
                    if algo == "fir":
                        modul.fir = ModAlgoAllModels(name, signal, time, algo)
                    if algo == "abc":
                        modul.abc = ModAlgoAllModels(name, signal, time, algo)
                    if algo == "spider monkey":
                        modul.spider_monkey = ModAlgoAllModels(name, signal, time, algo)
                self.modules.append(modul)