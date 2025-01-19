#Filename: node_layer.py
#Here the NodeLayer is beeing defined.

from network_layer import NetworkLayer
from node import Node

#The node type is either OUTPUT OR NODE
class NodeLayer(NetworkLayer):
    def __init__(self, nodes, previous_layer: NetworkLayer, output=False):
        super().__init__()
        self.type = "OUTPUT" if output else "NODE"
        self.nodes = [] #nodes in this layer
        self.previous_layer = previous_layer
        self.previous_layer.next_layer = self

        for i in range(nodes):
            if output == True:
                self.add_node(1)
            else: 
                self.add_node(0)

        # Connection of the nodes in this layer to the other nodes in the previous layer
        for own_node in self.nodes:
            for previous_node in self.previous_layer.nodes:
                own_node.connect_to(previous_node)

    def add_node(self, activation_function):
        self.nodes.append(Node(activation_function))

    def calculate_output(self):
        for node in self.nodes:
            node.calculate_value()

    def mutate(self):
        for node in self.nodes:
            node.mutate()

    def forward_propagation(self):
        self.calculate_output()
        if self.type == "OUTPUT":
            return [node.value for node in self.nodes]
        else:
            return self.next_layer.forward_propagation()


