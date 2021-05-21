import math
import numpy as np
def sigmoid(v):
    return 1 / (1 + math.exp(-v))

def hyperbolicTangentSigmoid(v):
    return (1 - math.exp(-v)) / (1 + math.exp(-v))


def forward_step(X, numOfOutputNeurons, isBiased, activationFunction, layersNum, neuronsDistribution):

    # FX & weight matrix
    FX = [None]*(layersNum+1)
    W = [None] * (layersNum + 1)

    # for each hidden layer
    for hl in range(layersNum):
        W[hl] = [None]*neuronsDistribution[hl]
        FX[hl] = [None]*neuronsDistribution[hl]
        FX[hl] = np.array(FX[hl])

        # for each neuron
        for n in range(neuronsDistribution[hl]):
            W[hl][n] = [0.0]*X.shape[0]
            W[hl][n] = np.array(W[hl][n])
            W[hl][n] = np.random.rand(1, X.shape[0])
            if not isBiased:
                W[hl][n][:, 0] = 0
            v = np.dot(W[hl][n], X)
            FX[hl][n] = sigmoid(v) if activationFunction == 'Sigmoid' else hyperbolicTangentSigmoid(v)

        X = FX[hl]

    # adding ones on FX for bias
    FX[hl] = np.reshape(FX[hl], (1, FX[hl].shape[0]))
    FX[hl] = np.c_[np.ones((FX[hl].shape[0], 1)), FX[hl]]
    hl += 1
    W[hl] = [None] * numOfOutputNeurons
    FX[hl] = [None] * numOfOutputNeurons
    FX[hl] = np.array(FX[hl])
    # for output layer neurons
    for n in range(numOfOutputNeurons):
        W[hl][n] = [0.0] * FX[hl-1].shape[1]
        W[hl][n] = np.array(W[hl][n])
        W[hl][n] = np.random.rand(1, FX[hl-1].shape[1])
        if not isBiased:
            W[layersNum][n][:, 0] = 0
        v = np.dot(W[hl][n], FX[hl-1].T)
        FX[hl][n] = sigmoid(v) if activationFunction == 'Sigmoid' else hyperbolicTangentSigmoid(v)

    # return FX of the output layer (y_pred)
    return W, FX


def backwardStep(y_train, W, FX, numOfOutputNeurons, layersNum, neuronsDistribution):
    y_pred = FX[layersNum]
    # Error of output layer
    outputError = (y_train-y_pred)*(y_pred)*(1 - y_pred)

    E = []
    # Error of hidden layers
    for hl in range(layersNum):
        for n in range(neuronsDistribution[hl]):
            e = outputError * W[layersNum-hl][n] * FX[hl][n] * (1 - FX[hl][n])
            E.append(e)

    return E


def updateWeights(W, learningRate, E, X, FX, layersNum, neuronsDistribution):
    for l in range(layersNum+1):
        for n in range(neuronsDistribution[l]):
            W[l][n] = W[l][n] + learningRate + E[l]
            print("")


def train(x_train, y_train, isBiased, learningRate, epochNum, layersNum, neuronsDistribution, activationFunction):

    # 1- add bias vector and create random weight vector w
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    # y_train = np.expand_dims(y_train, axis=1)



    for epoch in range(epochNum):
        for sample in range(x_train.shape[0]):
            X = x_train[sample]  # X vector of each sample
            Y = y_train[sample]
            # W is the matrix of weights for each neuron
            # FX is the matrix of net values after activation Fn for each neuron
            W, FX = forward_step(X, y_train.shape[1], isBiased, activationFunction, layersNum, neuronsDistribution)
            E = backwardStep(Y, W, FX, y_train.shape[1], layersNum, neuronsDistribution)
            W = updateWeights(W, learningRate, E, X, FX, layersNum, neuronsDistribution)
            print("")

def test(x_test, y_test):
    pass