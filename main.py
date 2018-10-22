from collect import *
from activations import *
from network import *
import sys
import numpy as np

#initialize layer sizes as list
layers = [784,3,4,10]

#initialize learning rate
learning_rate = 0.01

#initialize mini batch size
mini_batch_size = 16

#initialize epoch
epochs = 10

# initialize training, validation and testing data
training_data, validation_data, test_data = load_mnist()

#initialize neuralnet
nn = NeuralNetwork(layers, learning_rate, mini_batch_size, epochs)

#training neural network
nn.fit(training_data, validation_data)

#testing neural network
accuracy = nn.validate(test_data) / 100.0
print("Test Accuracy: " + str(accuracy) + "%")

#save the model
nn.save()
