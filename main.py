import pickle
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

population_size = 40
generation_size = 100

apple = pd.read_csv('Apple.csv')
apple = apple.replace([np.inf, -np.inf], np.nan)
apple = apple.dropna()
y = apple['Close']
x = apple.drop(columns=['Subtracao','Date','Open2','Close2'])
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1)

def calculate_fitness(population):
    global X_train, y_train, X_test
    fitness = []
    for idx, individual in enumerate(population):
        try:
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
            fitness.append((individual , mean_squared_error(y_test, y_pred)))
        except:
            fitness.append((individual, float('inf')))
    return fitness

def spin_wheel(roleta, sorted_probability):
    for idx, probability in enumerate(roleta):
        if idx > 0:
            if(sorted_probability <= probability and sorted_probability > roleta[idx-1]):
                break
        else:
            if(sorted_probability <= probability):
                break
    return idx

def parent_selection(population):
    #parents = select_random_parents(population)
    total_fitness = reduce(lambda x,y: x + (1/y[1]), population,0)
    parents = population
    roleta = []
    current_probability=0
    selected_parents = []
    parents.sort(key=lambda tup: tup[1], reverse=True)
    for parent in parents:
        roleta.append(current_probability + ((1/parent[1])/total_fitness))
        current_probability = roleta[-1]
    first_parent_id = spin_wheel(roleta, random.random())
    selected_parents.append(parents[first_parent_id])
    second_parent_id = spin_wheel(roleta, random.random())
    selected_parents.append(parents[second_parent_id])
    while(first_parent_id == second_parent_id):
        second_parent_id =spin_wheel(roleta, random.random())
        selected_parents[1] = parents[second_parent_id]    

    return list(map(lambda tup: tup[0], selected_parents)) #2 parents

def mutate(individual):
    params = [
        ('hidden_layer_sizes', random.randint(1, 20)), # hidden_layer_sizes - int
        ('activation', random.randint(0, 3)), # activation - object
        ('solver', random.randint(0, 2)), # solver - object
        ('batch_size', random.randint(1, 20)), # batch_size - int
        ('learning_rate', random.randint(0, 2)), # learning_rate - object
        ('max_iter', random.randint(10, 500)), # max_iter - int
        ('shuffle', random.randint(0, 1)), # shuffle - bool
        ('warm_start', random.randint(0, 1)), # warm_start - bool
        ('early_stopping', random.randint(0, 1)), # early_stopping - bool
        ('n_iter_no_change', random.randint(10, 30)) # n_iter_no_change - int
    ]
    param_mutation_idx = random.randint(0,9)
    key = params[param_mutation_idx][0]
    value = params[param_mutation_idx][1]
    individual[key] = value
    return individual

def recombine(parents):
    first_child = {}
    second_child = {}
    cut_idx = random.randint(0, 9)
    keys = [
        'hidden_layer_sizes',
        'activation',
        'solver',
        'batch_size',
        'learning_rate',
        'max_iter',
        'shuffle',
        'warm_start',
        'early_stopping',
        'n_iter_no_change'
    ]
    for idx, i in enumerate(keys):
        if idx <= cut_idx:
            first_child[i] = parents[0][i]
            second_child[i] = parents[1][i]
        else:
            first_child[i] = parents[1][i]
            second_child[i] = parents[0][i]
    return [first_child, second_child]

def generate_child():
    child = {
        'hidden_layer_sizes': random.randint(1, 20), # hidden_layer_sizes - int
        'activation': random.randint(0, 3), # activation - object
        'solver': random.randint(0, 2), # solver - object
        'batch_size': random.randint(1, 20), # batch_size - int
        'learning_rate': random.randint(0, 2), # learning_rate - object
        'max_iter': random.randint(10, 500), # max_iter - int
        'shuffle': random.randint(0, 1), # shuffle - bool
        'warm_start': random.randint(0, 1), # warm_start - bool
        'early_stopping': random.randint(0, 1), # early_stopping - bool
        'n_iter_no_change': random.randint(10, 30) # n_iter_no_change - int
    }
    return child

def survival_selection(population, parents):
    population.sort(key=lambda tup: tup[1], reverse=True)
    return population[2:]

def init_population():
    population = []
    for _ in range(population_size):
        population.append(generate_child())
    return population

def eval(population_fitness):
    for individual in population_fitness:
        if individual[1] == 0:
            return individual[0]
    return None

if __name__ == '__main__':
    f = open("roleta_output", "w")
    arquivo = open('roleta_pickle_out', 'wb')
    population = init_population()
    population_fitness = calculate_fitness(population)
    # population_fitness = list(map(lambda x: (x,0.1), population))
    solution = eval(population_fitness)
    count = 0
    while count < generation_size and solution == None:
        print("Generation", count)
        parents = parent_selection(population_fitness)
        children = recombine(parents)
        children = list(map(lambda child: mutate(child) if random.random() <= 0.4 else child, children)) 
        children = calculate_fitness(children)
        population_fitness.append(children[0])
        population_fitness.append(children[1])
        population_fitness = survival_selection(population_fitness, parents)
        solution = eval(population_fitness)
        # print(list(map(lambda x: x[1], population_fitness)))
        pickle.dump({f'Generation {count}': list(map(lambda x: x[1], population_fitness))}, arquivo)
        f.write(f"Generation {count}\n")
        f.write(f'[{", ".join(str(e) for e in list(map(lambda x: x[1], population_fitness)))}]\n\n')
        count += 1
    if count == generation_size:
        print(-1)
    else:
        total_converged = len(list(filter(lambda x : x[1] == 0, population_fitness)))
        print(count, total_converged)
    arquivo.close()
    f.close()
