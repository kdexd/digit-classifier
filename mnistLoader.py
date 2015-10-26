import cPickle
import gzip

import numpy as np



def loadData():
    file = gzip.open("/data/mnist.pkl.gz")
    trainingData, validationData, testData = cPickle.load(file)
    file.close()
    return trainingData, validationData, testData


def loadDataWrapper():
    tempTrainingData, tempValidationData, tempTestData = loadData()

    trainingInputs = [np.reshape(x, (784, 1)) for x in tempTrainingData[0]]
    trainingResults = [vectorizedResult(y) for y in tempTrainingData[1]]
    trainingData = zip(trainingInputs, trainingResults)

    validationInputs = [np.reshape(x, (784, 1)) for x in tempValidationData[0]]
    validationData = zip(validationInputs, tempTrainingData[1])

    testInputs = [np.reshape(x, (784, 1)) for x in tempTestData[0]]
    testData = zip(testInputs, tempTestData[1])

    return trainingData, validationData, testData


def vectorizedResult(j):
    e = np.zeros((10,1))
    e[j] = 1.0

    return e