from scipy.signal import savgol_filter

from dipl.conf import DIR
from dipl.data.search_space import to_model
from dipl.data.statistics import calculate_r2, calculate_spearman, calculate_nrmse
from dipl.evolutionary.MSO import MSO
from dipl.evolutionary.ant_colony import ant_colony_optimization
from dipl.evolutionary.atrificial_bee_colony import abc
from dipl.evolutionary.cuckoo_search import cuckoo_search
import matplotlib.pyplot as plt

from dipl.evolutionary.de import de_algorithm
from dipl.evolutionary.firefly_algorithm import firefly_algorithm
from dipl.evolutionary.ga import ga_algorithm
from dipl.evolutionary.pso import pso_algorithm


class ModAlgoAllModels:
    title: str
    signal: list
    time: list
    algorithm: str
    outputs: dict = {}
    models: list = ["lognormal", "gamma", "ldrw", "fpt", "lagged"]
    r_2: dict = {}
    spearman: dict = {}
    nrmse: dict = {}

    def run(self):

        # Library of models

        if self.algorithm == "pso":
            for model in self.models:
                print("Running " + model + " model")
                params = pso_algorithm(signal=self.signal,
                                       time=self.time,
                                       objective_function_type=model
                                       )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "de":
            for model in self.models:
                print("Running " + model + " model")
                params = de_algorithm(signal=self.signal,
                                      time=self.time,
                                      objective_function_type=model
                                      )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "ga":
            for model in self.models:
                print("Running " + model + " model")
                params = ga_algorithm(signal=self.signal,
                                      time=self.time,
                                      objective_function_type=model
                                      )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        # Own Evolutionary algorithms

        if self.algorithm == "ant":
            for model in self.models:
                print("Running " + model + " model")
                params = ant_colony_optimization(signal=self.signal,
                                                 time=self.time,
                                                 objective_function_type=model
                                                 )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "cuckoo search":
            for model in self.models:
                print("Running " + model + " model")
                params = cuckoo_search(signal=self.signal,
                                       time=self.time,
                                       objective_function_type=model
                                       )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "fir":
            for model in self.models:
                print("Running " + model + " model")
                params = firefly_algorithm(signal=self.signal,
                                           time=self.time,
                                           objective_function_type=model
                                           )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "abc":
            for model in self.models:
                print("Running " + model + " model")
                params = abc(signal=self.signal,
                             time=self.time,
                             objective_function_type=model
                             )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "spider monkey":
            for model in self.models:
                print("Running " + model + " model")
                params = MSO(signal=self.signal,
                             time=self.time,
                             objective_function_type=model).solve(100)
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

    def calc_r2(self):
        r2 = {}
        for model, fit in self.outputs.items():
            r2[model] = calculate_r2(self.signal, fit)
        self.r_2 = r2
        return r2

    def calc_spermans(self):
        spermans = {}
        for model, fit in self.outputs.items():
            spermans[model] = calculate_spearman(self.signal, fit)
        self.spearman = spermans
        return spermans

    def calc_nrmse(self):
        nrmse = {}
        for model, fit in self.outputs.items():
            nrmse[model] = calculate_nrmse(self.signal, fit)
        self.nrmse = nrmse
        return nrmse

    def show_statistics(self):
        print("R2: ", self.r_2)
        print("Spearman: ", self.spearman)
        print("NRMSE: ", self.nrmse)

    def plot(self):
        plt.figure()
        plt.title(self.title)
        plt.plot(self.time, self.signal, 'o', label='data')
        for model, fit in self.outputs.items():
            plt.plot(self.time, fit, '-', label=model)
        plt.legend(loc='upper right')
        plt.ylabel('Intensity')
        plt.xlabel('Time [s]')
        plt.show()
        plt.savefig(DIR + '\\data\\images\\' + self.algorithm + "_"
                    + self.title.replace(".mat", "")
                    + '.png')

    def __init__(self, title, signal, time, algorithm):
        self.title = title + " " + algorithm
        self.signal = signal
        self.time = time
        self.algorithm = algorithm
        self.run()
        self.r_2 = self.calc_r2()
        self.spearman = self.calc_spermans()
        self.nrmse = self.calc_nrmse()
        self.plot()
