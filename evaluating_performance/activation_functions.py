#Filename: activation_functions.py
#Here a few standard activation functions are defined.

import math
import numpy as np

class ActivationFunction:

    @staticmethod
    def ReLU(value: float) -> float:
        if value > 0:
            return value
        else:
            return 0

    #Prior I used the Sigmoid activation function, however because of the exponential component
    #and the lange input of cars, the numbers got too big. Instead I used SoftSign.
    @staticmethod
    def Sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))

    @staticmethod
    def Tanh(value: float) -> float:
        return (math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value))

    #Used alternative to sigmoid. Because of the better compability with large numbers.
    @staticmethod
    def SoftSign(value: float) -> float:
        return value / (1+np.absolute(value))
