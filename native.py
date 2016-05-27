import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.zs = [np.random.randn(y, 1) for y in sizes[1:]]
        self.activations = [np.random.randn(y, 1) for y in sizes[1:]]

