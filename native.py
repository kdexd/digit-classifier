import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork(object):
    def __init__(self, sizes):
        # Input layer is layer 0, followed by hidden layers layer 1, 2, 3...
        self.sizes = sizes
        self.num_layers = len(sizes)

        # First term corresponds to layer 0 (input layer). No weights enter the
        # input layer and hence self.weights[0] is redundant.
        self.weights = [[0], [np.random.randn(y, x)
                              for x, y in zip(sizes[1:], sizes[:-1])]]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self.zs = [np.random.randn(y, 1) for y in sizes]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self.activations = [np.random.randn(y, 1) for y in sizes]
