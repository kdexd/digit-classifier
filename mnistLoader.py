import cPickle
import gzip


def loadData():
    file = gzip.open("/data/mnist.pkl.gz")
    trainingData, validationData, testData = cPickle.load(file)
    file.close()
    return trainingData, validationData, testData
