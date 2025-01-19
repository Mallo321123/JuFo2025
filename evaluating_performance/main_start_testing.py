#filename: main_start_testing.py
#Run this file to test the diffrent algorithms

import numpy as np
import random
import pickle
from tqdm import tqdm 
from neural_network import NeuralNetwork
from fitness_function import fitness_score
from traffic_simulation import execute_simulation_nn, execute_simulation_algt, execute_simulation_algs

#Imports a saved model from the saved_model.pickle file.
def get_saved_model():
    with open('/home/lwcbwhite/JuFo_saves/JuFo/evaluating_performance/saved_model.pickle','rb') as file:
        neural_network = pickle.load(file)
    return neural_network

def start_testing(neural_network):

    #Setting how many cars appear per timestep on average.
    #Because this a testing enviorment the value should remain a constant for all the algorithms. 
    #Traffic and disapering traffic are varied by the poisson function for realism.
    def def_traffic_at_A():
        #training_values
        #traffic_at_A = random.uniform(0.1, 0.3) 
        traffic_at_A = 0.2
        return traffic_at_A 
    
    def def_traffic_at_B():
        #training_values
        #traffic_at_B = random.uniform(0.1, 0.3)
        traffic_at_B = 0.1
        return traffic_at_B 
    
    #Setting the duration of the testing simulation
    def def_steps():
        steps = 500
        return steps 

    def def_disapearing_traffic():
        d_traffic = 0.2
        return d_traffic

    #Testing the neural_network
    fitness_nn = execute_simulation_nn(def_traffic_at_A(), def_traffic_at_B(), def_disapearing_traffic(), def_disapearing_traffic(), def_steps(), neural_network, show=True)
    print('Fitness:', fitness_nn)
    print("\n")

    #Controll: regular timed algorithm 
    fitness_algt = execute_simulation_algt(def_traffic_at_A(), def_traffic_at_B(), def_disapearing_traffic(), def_disapearing_traffic(), def_steps(), show=True)
    print('Fitness:', fitness_algt)
    print("\n")

    #Comparission: regular timed algorithm wirh switching after one side has cleared
    fitness_algs = execute_simulation_algs(def_traffic_at_A(), def_traffic_at_B(), def_disapearing_traffic(), def_disapearing_traffic(), def_steps(), show=True)
    print('Fitness:', fitness_algs)
    print("\n")


neural_network = get_saved_model()
start_testing(neural_network)
