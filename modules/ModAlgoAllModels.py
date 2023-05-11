from dipl.data.search_space import to_model
from dipl.evolutionary.ant_colony import ant_colony_optimization
from dipl.evolutionary.atrificial_bee_colony import abc
from dipl.evolutionary.cuckoo_search_v2 import cuckoo_search
import matplotlib.pyplot as plt

from dipl.evolutionary.firefly_algorithm import firefly_algorithm
from dipl.evolutionary.pso import pso_algorithm


class ModAlgoAllModels:
    title: str
    signal: list
    time: list
    algorithm: str
    outputs: dict = {}
    models: list = ["gamma", "ldrw", "fpt", "lognormal", "lagged"]

    def run(self):
        if self.algorithm == "cuckoo search":
            for model in self.models:
                print("Running " + model + " model")
                out = cuckoo_search(signal=self.signal,
                                    time=self.time,
                                    objective_function_type=model
                                    )
                self.outputs[model] = out

        if self.algorithm == "pso":
            for model in self.models:
                print("Running " + model + " model")
                params = pso_algorithm(signal=self.signal,
                                       time=self.time,
                                       objective_function_type=model
                                       )
                out = to_model(signal=self.signal, time=self.time, model=model, params=params)
                self.outputs[model] = out

        if self.algorithm == "ant":
            for model in self.models:
                print("Running " + model + " model")
                params = ant_colony_optimization(signal=self.signal,
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

    def __init__(self, title, signal, time, algorithm):
        self.title = title + " " + algorithm
        self.signal = signal
        self.time = time
        self.algorithm = algorithm
        self.run()
        self.plot()
