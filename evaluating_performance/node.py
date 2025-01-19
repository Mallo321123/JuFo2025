#Filename: node.py
#Here a 'standard' node is defined as well as the mutation

import random
from activation_functions import ActivationFunction

class Node:
    def __init__(self, activation_function):
        self.previous_nodes = []
        self.value = 0
        self.weights = []
        #self.bias = random.uniform(-1,1)
        self.bias = 0
        self.activation_function = activation_function

    #Here the node connections are established. 
    #If there is no assigned weight, a random one is generated.
    def connect_to(self, node, weight=None):
        self.previous_nodes.append(node)
        if weight is None:
            self.weights.append(random.uniform(-2, 2))
        else:
            self.weights.append(weight)
        # print(self.weights)    

    #Returns the recalculated value of the node
    def calculate_value(self):
        v = 0
        for node in self.previous_nodes:
            v += node.value * self.weights[self.previous_nodes.index(node)]
        v += self.bias
        #Here the used activation function can be changed. E.g. to ReLu if needed.
        if self.activation_function == 0:
            v = ActivationFunction.ReLU(v)
        elif self.activation_function == 1:
            v = ActivationFunction.SoftSign(v)
        self.value = v
        return v

    #(random) mutation of the connections weights 
    # Choosing the right mutation rate and values is critical. 
    # If they were to big, you would overstep local maxima in the fitness landscape.
    # If they were to small, training would be inefficient and time intensive. 
    def mutate(self):
        for i in range(len(self.weights)):
            if random.random() < 0.25:
                if random.random() < 0.25:
                    #Setting a new weight between -1 and 1.
                    self.weights[i] = random.uniform(-1, 1)
                else:
                    #Changing the value of the weight by -0.25 to 0.25
                    self.weights[i] += random.uniform(-0.25, 0.25)
    
    def get_bias(self):
        return self.bias
