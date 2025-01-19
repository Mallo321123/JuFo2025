import numpy as np
import random
import copy
from tqdm import tqdm 
from fitness_function import fitness_score
from neural_network import NeuralNetwork
from traffic_simulation import execute_simulation   
from save_networks import save_model
from plot_fitness import plot_fitness_evolution

def start_training(neural_network, fitness_threshold):

    #To create a diverse neurnal network, which adapts to many traffic simulations, 
    #a varing traffic distrubution and intensity are used.
    def def_traffic_at_A():
        traffic_at_A = random.uniform(0.1, 0.3) #0.01 - 1
        return traffic_at_A 
    
    def def_traffic_at_B():
        traffic_at_B = random.uniform(0.1, 0.3)
        return traffic_at_B 
    
    def def_steps():
        steps = 200
        return steps 
    
    total_fitness = []
    #Increasing the number of traffic simulations increases the accuracy and stability of the fitness score.
    #However will also increases runtime exponetionally.
    traffic_situations = 1

    def def_disapearing_traffic():
        d_traffic = 0.2
        return d_traffic

    for j in range(traffic_situations):
        good_show = False
        avg_fitness_of_situation = execute_simulation(def_traffic_at_A(), def_traffic_at_B(), def_disapearing_traffic(), def_disapearing_traffic(), def_steps(), neural_network, good_show)
        if avg_fitness_of_situation >= fitness_threshold:
            good_show = True
            execute_simulation(def_traffic_at_A(), def_traffic_at_B(),def_disapearing_traffic(), def_disapearing_traffic(), def_steps(), neural_network, good_show)
            #neural_network.save_model()
        total_fitness.append(avg_fitness_of_situation)
    avg_total_fitness = np.average(total_fitness)
    return avg_total_fitness

#Getting the best models based on the given percentage. They are directly transferred to the next generation of training without being mutated.
def get_best_models(generational_fitness, generation_size, top_percentage, all_networks):

    best_models = []
    best_fitnesses = []

    for fitness in generational_fitness:
        if len(best_fitnesses) == 0:
            best_fitnesses.append(fitness)
        else:
            index = len(best_fitnesses)
            for sorted_fitness in best_fitnesses:
                if fitness > sorted_fitness:
                    index = best_fitnesses.index(sorted_fitness)
                    break
            best_fitnesses.insert(index, fitness)

    delete_count = int(generation_size * (1 - top_percentage))
    for x in range(delete_count):
        best_fitnesses.pop()

    for bf in best_fitnesses:
        best_models.append(copy.deepcopy(all_networks[generational_fitness.index(bf)]))
        
    return best_models

#The models are trained. Based on the fitness score the best models move onto the next generation. Aditionally, the rest of the new generation is
#created by choosing random models from the previous one and mutation them.
def train_generation(generation_size,all_networks_current,generational_fitness_current, all_networks_previous, generational_fitness_previous, fitness_threshold):
    for i in range(generation_size):
        generational_fitness_current.append(start_training(all_networks_current[i], fitness_threshold))
    
    current_copy = copy.deepcopy(all_networks_current)

    all_networks_previous.clear()
    for c in current_copy:
        all_networks_previous.append(c)

    generational_fitness_previous.clear()
    for f in generational_fitness_current:
        generational_fitness_previous.append(f)

    best_models = get_best_models(generational_fitness_current, generation_size, top_percentage, all_networks_current) 
    best_models_copy = copy.deepcopy(best_models)
    save_model(best_models[0])

    all_networks_current.clear()
    for c in best_models_copy:
        all_networks_current.append(c)
    
    for i in range(generation_size - len(best_models)):
        copyNetwork = copy.deepcopy(all_networks_previous[random.randint(0, len(all_networks_previous) - 1)])
        copyNetwork.mutate()
        all_networks_current.append(copyNetwork)
    
    generational_fitness_current.clear()
    

all_networks_current = []
all_networks_previous = []
generation_size = 100
generational_fitness_current = []
generational_fitness_previous = []

average_fitness_of_generation =[]
best_fitness_of_generation = []

#The top percentage determines how many models are tranfered to the next generation
#and how many are mutated.
top_percentage = 0.3

#Only plots and visualises models that exeed the fitness threshold
fitness_threshold = 12

generations = 75

#Here the starting generation is defined.
#The model uses three inputs: Traffic at the side A, Traffic at the side B, current runtime of signal phase.
for k in range(generation_size):
    neural_network = NeuralNetwork()
    neural_network.add_node_layer(3)
    neural_network.add_node_layer(8)
    neural_network.add_node_layer(6)
    neural_network.add_node_layer(4)
    neural_network.add_node_layer(1,True)

    all_networks_current.append(neural_network)


for i in tqdm(range(generations),total=generations):
    print("Generation " + str(i) + ": ")
    train_generation(generation_size,all_networks_current,generational_fitness_current, all_networks_previous, generational_fitness_previous, fitness_threshold)

    avg = 0.0
    max = 0.0

    for f in generational_fitness_previous:
        avg += f
        if f > max:
            max = f
    avg /= len(generational_fitness_previous)    

    print("Average fitness of generation " + str(i) + ": " + str(avg))
    print("Best fitness of generation " + str(i) + ": " + str(max))

    average_fitness_of_generation.append(avg)
    best_fitness_of_generation.append(max)

plot_fitness_evolution(average_fitness_of_generation, best_fitness_of_generation, generations)