import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def hyperbolicTangentSigmoid(v):
    t1 = (np.exp(-v) - np.exp(-v))
    return  t1/ (np.exp(-v) + np.exp(-v))


def forward_step(X, numOfOutputNeurons, isBiased, activationFunction, layersNum, neuronsDistribution, forUpdate=False, W=None):
    # --* FX is the output of activation Fn for each neuron
    # --* W is the weight matrix contains weight enters each neuron
    FX = [None] * (layersNum + 1)
    if not forUpdate:
        W = [None] * (layersNum + 1)

    # for each hidden layer
    for l in range(layersNum + 1):
        if l == 0:
            # if input layer
            lastNeuronsNum = X.shape[0]
        else:
            lastNeuronsNum = neuronsDistribution[l - 1] + 1  # this '+1' is for bias

        if l == layersNum:
            # if output layer
            currentNeuronsNum = numOfOutputNeurons
        else:
            currentNeuronsNum = neuronsDistribution[l] + 1  # this '+1' is for bias
        if not forUpdate:
            W[l] = np.random.rand(currentNeuronsNum, lastNeuronsNum)
            if not isBiased:
                W[l][:, 0] = 0

        # net value
        v = np.dot(W[l], X)
        FX[l] = sigmoid(v) if activationFunction == 'Sigmoid' else hyperbolicTangentSigmoid(v)
        if l != layersNum:
            FX[l][0] = 1  # bias of the next layer
        FX[l] = np.reshape(FX[l], (currentNeuronsNum, 1))
        X = FX[l]

    return W, FX


def forward_step_Testing(W, X, numOfOutputNeurons, activationFunction, layersNum, neuronsDistribution):
    FX = [None] * (layersNum + 1)
    # for each hidden layer
    for l in range(layersNum + 1):
        if l == layersNum:
            currentNeuronsNum = numOfOutputNeurons
        else:
            currentNeuronsNum = neuronsDistribution[l]+1

        # net value
        v = np.dot(W[l], X)
        FX[l] = sigmoid(v) if activationFunction == 'Sigmoid' else hyperbolicTangentSigmoid(v)
        FX[l] = np.reshape(FX[l], (currentNeuronsNum, 1))

        X = FX[l]
    return FX[layersNum]


def backwardStep(y_train, W, FX, layersNum):
    y_pred = FX[layersNum]
    E = [None] * (layersNum + 1)
    # Error of output layer
    E[layersNum] = (y_train - y_pred) * (y_pred * (1 - y_pred))
    # Error of hidden layers
    for l in range(layersNum - 1, -1, -1):
        E[l] = (np.dot(np.transpose(W[l + 1]), E[l + 1])) * (FX[l] * (1 - FX[l]))

    # error matrix that contains error at each neuron
    return E


def updateWeights(W, learningRate, E, X, FX, layersNum):
    # update weight at each neuron starting from HL 1
    for l in range(layersNum + 1):
        if l == 0:
            W[l] = W[l] + learningRate * np.dot(E[l], np.transpose(X))
        else:
            W[l] = W[l] + learningRate * np.dot(E[l], np.transpose(FX[l - 1]))
    return W

'''


'''
def train(x_train, y_train, isBiased, learningRate, epochNum, layersNum, neuronsDistribution, activationFunction):
    # 1- add bias vector and create random weight vector w
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    # y_train = np.expand_dims(y_train, axis=1)

    W = [None] * x_train.shape[0]
    for epoch in range(epochNum):
        for sample in range(x_train.shape[0]):
            X = np.expand_dims(x_train[sample], axis=1)  # X vector of each sample
            Y = np.expand_dims(y_train[sample], axis=1)  # Y vector of each sample

            # W is the matrix of weights for each neuron
            # FX is the matrix of net values after activation Fn for each neuron
            W[sample], FX = forward_step(X, Y.shape[0], isBiased, activationFunction, layersNum, neuronsDistribution)
            while True:
                E = backwardStep(Y, W[sample], FX, layersNum)
                mse = np.sum(np.power(E[layersNum],2))
                if(mse <= 0.01):
                    break
                W[sample] = updateWeights(W[sample], learningRate, E, X, FX, layersNum)
                W[sample], FX = forward_step(X, Y.shape[0], isBiased, activationFunction, layersNum, neuronsDistribution, True, W[sample])


    return W


'''
    set the maximum value in predicted y with 1
    and all other values with 0
'''
def updateYDash(yDash):
    indexOfMax = np.argmax(yDash, axis=0)
    yDash[:, 0] = 0
    yDash[indexOfMax] = 1
    yDash = np.transpose(yDash)
    return  yDash


def test(x_test, y_test, isBiased, layersNum, neuronsDistribution, activationFunction, W):
    x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]
    y_pred = np.empty(y_test.shape)
    YConfusionMatrix=[]
    PConfusionMatrix=[]
    for sample in range(x_test.shape[0]):
        X = np.expand_dims(x_test[sample], axis=1)  # X vector of each sample
        Y = np.expand_dims(y_test[sample], axis=1)  # X vector of each sample
        yDash = forward_step_Testing(W[sample], X, Y.shape[0], activationFunction, layersNum, neuronsDistribution)
        Y = yDash
        MaxInd=0
        MaxVal=Y[0]
        for j in range(len(Y)):
            if Y[j]>MaxVal:
                MaxInd=j
                MaxVal=Y[j]
        PConfusionMatrix.append(MaxInd)
        y_pred[sample] = updateYDash(yDash)

    for i in range(len(y_test)):
        for j in range(len(y_test[0])):
            if y_test[i,j]==1:
                YConfusionMatrix.append(j)
                break
    NumOfMiss=0
    for i in range(len(PConfusionMatrix)):
            if(PConfusionMatrix[i]!=YConfusionMatrix[i]):
                NumOfMiss+=1

    accuracy = 100 - ((NumOfMiss / len(PConfusionMatrix)) * 100)
    if isBiased:
        print("======== Testing Accuracy with bias is : ", accuracy , "%")
    else:
        print("======== Testing Accuracy is : ", accuracy, "%")

    evaluate(YConfusionMatrix,PConfusionMatrix)


def evaluate(y_test, y_pred):
    # It should return the accuracy and show the confusion matrix
    labels = ['class 1', 'class 2', 'class 3']
    confusion_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(confusion_mat, classes=labels)
    acc = precision_score(y_test, y_pred, average='micro')
    print("The Accuracy from Confusion Matrix is : ", acc * 100, "%")


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    print(cm)
    plt.show()
