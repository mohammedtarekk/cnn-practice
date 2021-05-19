import math

def sigmoid(v):
    return 1 / (1 + math.exp(-v))

def hyperbolicTangentSigmoid(v):
    return (1 - math.exp(-v)) / (1 + math.exp(-v))

def train(x_train, y_train, isBiased, learningRate, epochNum, layersNum, neuronsDistribution, activationFunction):
    pass

def test(x_test, y_test):
    pass