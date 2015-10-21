import random
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


    def feedforward (self, activation):
        # output = sigmoid(activation of previous * weight of current + bias of current)

        for b, w in zip(self.biases, self.weights):
            # activation was passed as a parameter to have it as a list, not a number
            activation = self.sigmoid (np.dot(w, activation) + b)
        return activation



    def stochasticGradDesc(self, trainingData, epochs, miniBatchSize, eta, testData=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The "training_data" is a list of tuples
        "(x, y)" representing the training inputs and the desired
        outputs. The number of epochs to train for, and the size of
        mini-batches to use when sampling. If "test_data" is provided
        then network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        if testData:
            numberOfTests = len (testData)

        n = len(trainingData)

        for j in xrange(epochs):
            random.shuffle(trainingData)
            miniBatches = [trainingData[k:k+miniBatchSize] for k in xrange(0, n, miniBatchSize)]

            for miniBatch in miniBatches:
                self.updateMiniBatch(miniBatch, eta)

            if testData:
                print "Epoch {0}: {1} / {2}".format(j, self.evaluate(testData), numberOfTests)
            else:
                print "Epoch {0} complete".format(j)


    def updateMiniBatch (self, miniBatch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``miniBatch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]

        for x, y in miniBatch:
            deltaDelB, deltaDelW = self.backprop(x, y)
            delB = [nb+dnb for nb, dnb in zip(delB, deltaDelB)]
            delW = [nw+dnw for nw, dnw in zip(delW, deltaDelW)]
        self.weights = [w-(eta/len(miniBatch))*nw
                        for w, nw in zip(self.weights, delW)]
        self.biases = [b-(eta/len(miniBatch))*nb
                       for b, nb in zip(self.biases, delB)]
        
        
    def backprop(self, x, y):
        """Return a tuple ``(delB, delW)`` representing the
        gradient for the cost function C_x.  ``delB`` and
        ``delW`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        delB = [np.zeros(b.shape) for b in self.biases]
        delW = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.costDerivative(activations[-1], y) * \
            self.sigmoidPrime(zs[-1])
        delB[-1] = delta
        delW[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.numberOfLayers):
            z = zs[-l]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delB[-l] = delta
            delW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (delB, delW)


    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def costDerivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)