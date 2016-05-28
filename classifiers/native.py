import numpy as np
import random


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
        self.weights = [np.array([0])] + [np.random.randn(y, x)
                        for y, x in zip(sizes[1:], sizes[:-1])]

        # Input layer does not have any biases. self.biases[0] is redundant.
        self.biases = [np.random.randn(y, 1) for y in sizes]

        # Input layer has no weights, biases associated. Hence z = wx + b is not
        # defined for input layer. self.zs[0] is redundant.
        self.zs = [np.random.randn(y, 1) for y in sizes]

        # Training examples can be treated as activations coming out of input
        # layer. Hence self.activations[0] = (training_example).
        self.activations = [np.random.randn(y, 1) for y in sizes]

    def fit(self, training_data, mini_batch_size=1, epochs=1, eta=1.0):
        if mini_batch_size == 1:
            self.incremental_gradient_descent(training_data, epochs, eta)
        else:
            self.stochastic_gradient_descent(training_data, mini_batch_size,
                                             epochs, eta)

    def predict(self, x):
        self.feedforward(x)
        return np.argmax(self.activations[-1])

    def feedforward(self, x):
        self.activations[0] = x
        for i in range(1, self.num_layers):
            self.zs[i] = np.dot(self.weights[i], self.activations[i - 1]) + \
                         self.biases[i]
            self.activations[i] = sigmoid(self.zs[i])

    def back_propagation(self, x, y):
        # First a feed forward run.
        self.feedforward(x)

        # Initialization of matrices to hold errors.
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        # Calculate error and cost derivative for output layer.
        error = (self.activations[-1] - y) * sigmoid_prime(self.zs[-1])
        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error, self.activations[-2].transpose())

        # Backward pass of error.
        for l in range(self.num_layers - 2, 0, -1):
            error = np.dot(self.weights[l + 1].transpose(), error) * \
                sigmoid_prime(self.zs[l])

            nabla_b[l] = error
            nabla_w[l] = np.dot(error, self.activations[l - 1].transpose())

        return nabla_b, nabla_w

    def incremental_gradient_descent(self, training_data, epochs, eta):
        while epochs > 0:
            random.shuffle(training_data)
            # Incremental learning of neural network. Cost function is updated
            # after calculation of error from each example.
            for x, y in training_data:
                nabla_b, nabla_w = self.back_propagation(x, y)
                self.weights = [w - eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - eta * db for b, db in zip(self.biases, nabla_b)]
            epochs -= 1

    def stochastic_gradient_descent(self, training_data, mini_batch_size,
                                    epochs, eta):
        while epochs > 0:
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in
                            range(0, len(training_data), mini_batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(bias.shape) for bias in self.biases]
                nabla_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
                    nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [w - (eta / mini_batch_size) * dw for w, dw in
                                zip(self.weights, nabla_w)]
                self.biases = [b - (eta / mini_batch_size) * db for b, db in
                               zip(self.biases, nabla_b)]
            epochs -= 1
