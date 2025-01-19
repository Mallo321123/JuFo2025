#Filname: neural_network.py
#Here the neural network itself is defined and created.

from input_layer import InputLayer
from node_layer import NodeLayer
from network_layer import NetworkLayer
import json
import os

class NeuralNetwork (NetworkLayer):

    def __init__(self):
        self.layers = []

    #Adds a new layer to the network. If its the first layer of the network, an InputLayer is created automatically.
    #If output is set to true, the layer will be the last layer of the network
    def add_node_layer(self, nodes, output = None):
        n = len(self.layers)
        if n == 0:
            self.layers.append(InputLayer(nodes))
        else:
            if output is not None:
                self.layers.append(NodeLayer(nodes, self.layers[n-1], output))
            else:
                self.layers.append(NodeLayer(nodes, self.layers[n-1]))
            self.layers[n - 1].nextLayer = self.layers[n]
    
    #starts the forward-propagation of the network
    def think(self, input):
        return self.layers[0].forward_propagation(input)

    #starts the random mutation process of the network
    def mutate(self): 
        for k in range(len(self.layers)):
            self.layers[k].mutate()
        