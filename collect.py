import os
import gzip
import cPickle
import wget

import numpy as np


def load_mnist():
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))
        wget.download('http://deeplearning.net/data/mnist/mnist.pkl.gz', out='data')

    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
    training_data, validation_data, test_data = cPickle.load(data_file)
    data_file.close()

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [vectorized_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_results = validation_data[1]
    validation_data = zip(validation_inputs, validation_results)

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    return training_data, validation_data, test_data


def vectorized_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e
