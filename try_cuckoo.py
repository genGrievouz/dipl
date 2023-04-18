import math
import numpy as np
import matplotlib.pyplot as plt

from functions.load_data import load_data, preprocessing, normalize, remove_negative_values


# Load signal data
data = load_data()
signal_data = data

for k, v in data.items():
    print(k)
    x = normalize(remove_negative_values(preprocessing((data[k]['signal']))))
    ind = int(np.round(0.5 * len(x)))
    x = x[:ind]
    time = data[k]['time']
    time = time[:ind]
    ts = data[k]['ts']

# Define objective function
def objective_function(params):
    auc, alpha, beta = params
    model_output = [(auc * (i ** alpha) * np.exp(-1 * i / beta)) / (beta ** (alpha + 1) * math.gamma(alpha + 1)) for i in signal_data]
    # Compute difference between model output and signal data
    diff = np.subtract(model_output, signal_data)
    # Compute sum of squared differences as fitness value
    fitness = np.sum(diff**2)
    return fitness


def cuckoo_search_optimization(objective_func, x, n_cuckoos=10, n_iterations=100, auc_range=(0, 1), alpha_range=(0, 1), beta_range=(0, 1)):
    """
    Optimize function parameters using cuckoo search algorithm.
    """

    def get_random_params():
        """
        Generate random parameter values within given ranges.
        """
        auc = np.random.uniform(auc_range[0], auc_range[1])
        alpha = np.random.uniform(alpha_range[0], alpha_range[1])
        beta = np.random.uniform(beta_range[0], beta_range[1])
        return auc, alpha, beta

    def get_fitness(params):
        """
        Evaluate fitness of parameter values using objective function.
        """
        auc, alpha, beta = params
        y = objective_func(x, auc, alpha, beta)
        return np.mean(y)

    # Initialize cuckoos with random parameter values
    cuckoos = [get_random_params() for _ in range(n_cuckoos)]
    best_fitness = float('-inf')
    best_params = None

    # Cuckoo search iterations
    for _ in range(n_iterations):
        for i in range(n_cuckoos):
            cuckoo = cuckoos[i]
            new_cuckoo = cuckoo
            # Generate a new cuckoo by performing random walk
            step_size = 0.01  # Step size for random walk
            for j in range(3):  # 3 parameters: auc, alpha, beta
                new_cuckoo[j] += step_size * np.random.randn()
                # Clamp parameter values within given ranges
                new_cuckoo[j] = np.clip(new_cuckoo[j], auc_range[0], auc_range[1])
                new_cuckoo[j] = np.clip(new_cuckoo[j], alpha_range[0], alpha_range[1])
                new_cuckoo[j] = np.clip(new_cuckoo[j], beta_range[0], beta_range[1])

            # Evaluate fitness of new cuckoo
            new_fitness = get_fitness(new_cuckoo)

            # Update cuckoo if better fitness is found
            if new_fitness > best_fitness:
                best_fitness = new_fitness
                best_params = new_cuckoo
                cuckoos[i] = new_cuckoo

    return best_params, best_fitness

# Define parameter ranges
auc_range = [0, 2]
alpha_range = [0, 2]
beta_range = [0, 2]
n_iterations = 100

# Run cuckoo search optimization
best_params, best_fitness = cuckoo_search_optimization(objective_function, x=signal_data, n_iterations=100, auc_range=auc_range, alpha_range=alpha_range, beta_range=beta_range)

# Extract the best parameters
best_auc, best_alpha, best_beta = best_params
print("Best AUC: ", best_auc)
print("Best Alpha: ", best_alpha)
print("Best Beta: ", best_beta)
print("Best Fitness: ", best_fitness)


# Generate fitted model output using best parameters
x = np.linspace(0, len(signal_data), len(signal_data))  # Assuming x-axis represents time or data points
fitted_model_output = [(best_auc * (i ** best_alpha) * np.exp(-1 * i / best_beta)) / (best_beta ** (best_alpha + 1) * math.gamma(best_alpha + 1)) for i in x]

# Plot the signal data and fitted model output
plt.figure(figsize=(8, 6))
plt.plot(x, signal_data, label='Signal Data')
plt.plot(x, fitted_model_output, label='Fitted Model Output')
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.legend()
plt.title('Signal Data vs Fitted Model Output')
plt.show()