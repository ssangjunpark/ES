import numpy as np
import matplotlib.pyplot as plt

class evolution_strategies:
    def __init__(self, genertation_size, population_size, parameters, learning_rate, noise):
        self.generation_size = genertation_size
        self.population_size = population_size
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.noise = noise

    def train(self, F):
        generation_average_fitness = np.empty(self.generation_size)
        param = self.parameters

        for generation_id in range(self.generation_size):
            
            # sample from standard normal
            eps = np.random.randn(self.population_size, len(self.parameters))
            generation_fitness = np.empty(self.population_size)

            #populate and evaluate fitness
            for population_id in range(self.population_size):
                param_individual = param + self.noise * eps[population_id]
                fitness = F(param_individual)
                generation_fitness[population_id] = fitness
            
            # update param and maximize fitness function
            standarized_generation_fitness = (generation_fitness - generation_fitness.mean()) / generation_fitness.std()
            param = param + self.learning_rate * (1/(self.population_size * self.noise)) * np.dot(eps.T, standarized_generation_fitness)

            generation_average_fitness[generation_id] = generation_fitness.mean()
        
        return param, generation_average_fitness

def F(parameters):
    # concave paraboloid
    return -(0.3 * (parameters[0] ** 2) + (2 * parameters[1])**2 + 0.7 * (parameters[2] + 7)**2)

if __name__ == "__main__":
    es = evolution_strategies(genertation_size=1000, population_size=50, parameters=np.random.randn(3), learning_rate=1e-3, noise=0.1)

    param, generation_average_fitness = es.train(F=F)

    plt.plot(generation_average_fitness)
    plt.show()

    print(param)







    
