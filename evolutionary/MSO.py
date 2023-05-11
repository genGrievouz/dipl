import numpy as np


class MSO:
    def __init__(self, fitness_func, num_variables, lb, ub, num_monkeys=30, num_spiders=10, spider_radius=0.2,
                 alpha=0.1, gamma=0.1, beta=2):
        self.fitness_func = fitness_func
        self.num_variables = num_variables
        self.lb = lb
        self.ub = ub
        self.num_monkeys = num_monkeys
        self.num_spiders = num_spiders
        self.spider_radius = spider_radius
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def solve(self, max_iter):
        # Initialize monkeys and spiders
        monkeys = np.random.uniform(self.lb, self.ub, size=(self.num_monkeys, self.num_variables))
        spiders = np.zeros((self.num_spiders, self.num_variables))

        # Evaluate initial fitness
        monkey_fitness = np.apply_along_axis(self.fitness_func, 1, monkeys)

        # Main loop
        for i in range(max_iter):
            # Sort monkeys by fitness
            sorted_indices = np.argsort(monkey_fitness)
            monkeys = monkeys[sorted_indices]
            monkey_fitness = monkey_fitness[sorted_indices]

            # Update best monkey and fitness
            best_monkey = monkeys[0]
            best_fitness = monkey_fitness[0]

            # Update spider positions
            for j in range(self.num_spiders):
                center = best_monkey + self.alpha * (best_monkey - monkeys[j])
                spiders[j] = center + self.spider_radius * np.random.uniform(low=-1, high=1, size=self.num_variables)

            # Evaluate spider fitness
            spider_fitness = np.apply_along_axis(self.fitness_func, 1, spiders)

            # Update monkey positions
            for j in range(self.num_monkeys):
                # Calculate attraction to best monkey
                delta = np.abs(best_monkey - monkeys[j])
                attractor = self.gamma * delta * np.random.uniform(low=-1, high=1, size=self.num_variables)

                # Calculate repulsion from spiders
                repulsors = np.zeros((self.num_spiders, self.num_variables))
                for k in range(self.num_spiders):
                    delta = np.abs(spiders[k] - monkeys[j])
                    repulsors[k] = self.beta * delta * np.random.uniform(low=-1, high=1, size=self.num_variables)
                repulsion = np.sum(repulsors, axis=0)

                # Update position
                monkeys[j] += attractor - repulsion

                # Check boundaries
                monkeys[j] = np.clip(monkeys[j], self.lb, self.ub)

            # Evaluate new fitness
            monkey_fitness = np.apply_along_axis(self.fitness_func, 1, monkeys)

            # Update best monkey and fitness
            if monkey_fitness[0] < best_fitness:
                best_monkey = monkeys[0]
                best_fitness = monkey_fitness[0]

            # Print progress
            print(f"Iteration {i + 1}/{max_iter}: Best fitness = {best_fitness}")

        print(best_monkey)
        return best_monkey, best_fitness
