"""
Wrapper module to save the weights and biases of a particular network model
as compressed numpy binaries.
"""
import os

import numpy as np

from network import NeuralNetwork

model_dirpath = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, 'models'
)


def save_model(network, dirpath=model_dirpath, filename='model.npz'):
    np.savez_compressed(
        file=os.path.join(dirpath, filename),
        weights=network.weights,
        biases=network.biases
    )


def model_from_file(filepath):
    npz_members = np.load(filepath)
    network = NeuralNetwork()
    network.from_npz(npz_members)
    return network
