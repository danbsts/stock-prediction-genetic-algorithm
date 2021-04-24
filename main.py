import numpy


import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

activation = ['identity', 'logistic', 'tanh', 'relu']
solver = ['lbfgs', 'sgd', 'adam']
learning_rate = ['constant', 'invscaling', 'adaptive']
bool_array = [False, True]

population_size = 40 # vai demorar 7 anos
generation_size = 100

apple = pd.read_csv('Apple.csv')
apple = apple.replace([np.inf, -np.inf], np.nan)
apple = apple.dropna()
y = apple['Close']
x = apple.drop(columns=['Adj Close', 'Date', 'Close', 'Volume'])
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

def calculate_fitness(population):
    global X_train, y_train, X_test
    fitness = []
    for idx, individual in enumerate(population):
        print('Filho', idx)
        regr = MLPRegressor(
            hidden_layer_sizes=individual['hidden_layer_sizes'],
            activation=activation[individual['activation']],
            solver=solver[individual['solver']],
            batch_size=individual['batch_size'],
            learning_rate=learning_rate[individual['learning_rate']],
            max_iter=individual['max_iter'],
            shuffle=bool_array[individual['shuffle']],
            warm_start=bool_array[individual['warm_start']],
            early_stopping=bool_array[individual['early_stopping']],
            n_iter_no_change=individual['n_iter_no_change']
        )
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test) #Fazendo a predição no conjunto de teste
        fitness.append(mean_squared_error(y_test, y_pred))
    return fitness

def generate_child():
    child = {
        'hidden_layer_sizes': random.randint(1, 20), # hidden_layer_sizes - int
        'activation': random.randint(0, 3), # activation - object
        'solver': random.randint(0, 2), # solver - object
        'batch_size': random.randint(1, 20), # batch_size - int
        'learning_rate': random.randint(0, 2), # learning_rate - object
        'max_iter': random.randint(10, 100), # max_iter - int
        'shuffle': random.randint(0, 1), # shuffle - bool
        'warm_start': random.randint(0, 1), # warm_start - bool
        'early_stopping': random.randint(0, 1), # early_stopping - bool
        'n_iter_no_change': random.randint(10, 30) # n_iter_no_change - int
    }
    return child

def init_population():
    population = []
    for _ in range(population_size):
        population.append(generate_child())
    return population

if __name__ == '__main__':
    print('bom dia casada')
    population = init_population()
    print('pop iniciada')
    population_fitness = calculate_fitness(population)
    print(population_fitness)
    # for _ in range(generation_size):
        # parents = parent_selection(population_fitness)
        # children = cut_and_crossfill(parents)
    #     children = list(map(lambda child: mutate(child) if random.random() <= 0.4 else child, children)) 
    #     children = calculate_fitness(children)
    #     population_fitness.append(children[0])
    #     population_fitness.append(children[1])
    #     population_fitness = survival_selection(population_fitness, parents)
    #     solution = eval(population_fitness)
    # if 10000:
    #     return -1
    # else:
    #     total_converged = len(list(filter(lambda x : x[1] == 1, population_fitness)))
    #     return (count, total_converged, calculate_mean(population_fitness,1), calculate_std(population_fitness,1))
