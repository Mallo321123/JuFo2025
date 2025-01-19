#Filename: input_layer.py
#Here the first layer of my neural network is defined.
#This layer represents the input layer and therefore has 
#neither weights nor mutations.
from network_layer import NetworkLayer
from node import Node

class InputLayer(NetworkLayer):
    def __init__(self, input_length):
        super().__init__()
        self.nodes = [Node(0) for j in range(input_length)]

    def calculate_output(self, input_data):
        #Values of the nodes in the input layer are being set to the input of the network
        for x in range(len(self.nodes)):
            self.nodes[x].value = input_data[x]

    def mutate(self):
        #The input layer doesn't have to mutate
        pass

    def forward_propagation(self, input_data):
        self.calculate_output(input_data)
        return self.next_layer.forward_propagation()

    def add_node(self):
        #Creates an new input node
        self.nodes.append(Node(0))

