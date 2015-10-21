import numpy as np

class NeuralNetwork (object):

    def __init__ (self, sizes):
        # sizes is a list of neurons in successive layers

        # the number of layers in the network is equal to length of 'sizes' list
        self.numberOfLayers = len(sizes)

        self.sizes = sizes

        # random initialization of weights/ biases for a starting to gradient descent
        # random generation is through guassian distribution
        # random numbers will have mean 0 and std dev 1
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


    def sigmoid(z):
        # if the input is numpy vector or matrix, sigmoid would be applied element-wise
        return 1.0/(1.0 + np.exp(-z))
