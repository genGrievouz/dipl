from dipl.data.search_space import get_objective_function_and_params
import numpy as np


def ant_colony_optimization(
        signal,
        time,
        objective_function_type,
):

    global gamma_values, gamma_probabilities, gamma_index, best_gamma
    params_range, objective_function = get_objective_function_and_params(signal, time, objective_function_type)

    # ACO parameters
    n_ants = 20
    n_iterations = 100
    decay = 0.1
    alpha = 1
    beta = 2

    n_params = len(params_range)
    # Parameter discretization
    auc_values = np.linspace(params_range[0][0], params_range[0][1], n_ants)
    beta_values = np.linspace(params_range[1][0], params_range[1][1], n_ants)
    alpha_values = np.linspace(params_range[2][0], params_range[2][1], n_ants)
    if n_params == 4:
        gamma_values = np.linspace(params_range[3][0], params_range[3][1], n_ants)

    # Pheromone maps
    pheromone_auc = np.ones(n_ants)
    pheromone_beta = np.ones(n_ants)
    pheromone_alpha = np.ones(n_ants)
    pheromone_gamma = np.ones(n_ants)

    # ACO main loop
    for i in range(n_iterations):
        # Initialize new generation of ants
        new_auc_values = np.zeros(n_ants)
        new_beta_values = np.zeros(n_ants)
        new_alpha_values = np.zeros(n_ants)
        new_gamma_values = np.zeros(n_ants)


        for ant in range(n_ants):
            # Transition rule
            auc_probabilities = (pheromone_auc ** alpha) * ((1.0 / auc_values) ** beta)
            auc_probabilities = auc_probabilities / auc_probabilities.sum()
            beta_probabilities = (pheromone_beta ** alpha) * ((1.0 / beta_values) ** beta)
            beta_probabilities = beta_probabilities / beta_probabilities.sum()
            alpha_probabilities = (pheromone_alpha ** alpha) * ((1.0 / alpha_values) ** beta)
            alpha_probabilities = alpha_probabilities / alpha_probabilities.sum()
            if n_params == 4:
                gamma_probabilities = (pheromone_alpha ** alpha) * ((1.0 / gamma_values) ** beta)
                gamma_probabilities = gamma_probabilities / gamma_probabilities.sum()

            # Assign new values
            new_auc_values[ant] = np.random.choice(auc_values, p=auc_probabilities)
            new_beta_values[ant] = np.random.choice(beta_values, p=beta_probabilities)
            new_alpha_values[ant] = np.random.choice(alpha_values, p=alpha_probabilities)
            if n_params == 4:
                new_gamma_values[ant] = np.random.choice(gamma_values, p=gamma_probabilities)

        # Update pheromones
        for ant in range(n_ants):
            auc_index = np.where(auc_values == new_auc_values[ant])[0][0]
            beta_index = np.where(beta_values == new_beta_values[ant])[0][0]
            alpha_index = np.where(alpha_values == new_alpha_values[ant])[0][0]
            if n_params == 4:
                gamma_index = np.where(gamma_values == new_gamma_values[ant])[0][0]
                fitness = np.mean(
                    objective_function(
                        (new_auc_values[ant], new_beta_values[ant], new_alpha_values[ant], new_gamma_values[ant]),
                        signal))
            if n_params == 3:
                fitness = np.mean(
                    objective_function((new_auc_values[ant], new_beta_values[ant], new_alpha_values[ant]), signal))
            pheromone_auc[auc_index] = (1 - decay) * pheromone_auc[auc_index] + decay * fitness
            pheromone_beta[beta_index] = (1 - decay) * pheromone_beta[beta_index] + decay * fitness
            pheromone_alpha[alpha_index] = (1 - decay) * pheromone_alpha[alpha_index] + decay * fitness
            if n_params == 4:
                pheromone_gamma[gamma_index] = (1 - decay) * pheromone_gamma[gamma_index] + decay * fitness

    # Final result
    best_auc = auc_values[pheromone_auc.argmax()]
    best_beta = beta_values[pheromone_beta.argmax()]
    best_alpha = alpha_values[pheromone_alpha.argmax()]
    if n_params == 4:
        best_gamma = gamma_values[pheromone_gamma.argmax()]

    if n_params == 4:
        return best_auc, best_beta, best_alpha, best_gamma

    return best_auc, best_beta, best_alpha
