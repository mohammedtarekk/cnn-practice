import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score

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

def forward_step_Testing(W,X, numOfOutputNeurons, isBiased, activationFunction, layersNum, neuronsDistribution):

    # FX & weight matrix
    FX = [None]*(layersNum+1)

    # for each hidden layer
    for hl in range(layersNum):
        FX[hl] = [None]*neuronsDistribution[hl]
        FX[hl] = np.array(FX[hl])

        # for each neuron
        for n in range(neuronsDistribution[hl]):
            v = np.dot(W[hl][n], X)
            FX[hl][n] = sigmoid(v) if activationFunction == 'Sigmoid' else hyperbolicTangentSigmoid(v)

        X = FX[hl]

    # adding ones on FX for bias
    FX[hl] = np.reshape(FX[hl], (1, FX[hl].shape[0]))
    FX[hl] = np.c_[np.ones((FX[hl].shape[0], 1)), FX[hl]]
    hl += 1
    FX[hl] = [None] * numOfOutputNeurons
    FX[hl] = np.array(FX[hl])
    # for output layer neurons
    for n in range(numOfOutputNeurons):
        v = np.dot(W[hl][n], FX[hl-1].T)
        FX[hl][n] = sigmoid(v) if activationFunction == 'Sigmoid' else hyperbolicTangentSigmoid(v)

    # return FX of the output layer (y_pred)
    return FX


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
            #E = backwardStep(Y, W, FX, y_train.shape[1], layersNum, neuronsDistribution)
            #W = updateWeights(W, learningRate, E, X, FX, layersNum, neuronsDistribution)
            print("")
    return W

def test(x_test, y_test,isBiased,learningRate,epochNum,layersNum,neuronsDistribution,activationFunction,W):
    x_test=np.c_[np.ones((x_test.shape[0], 1)), x_test]
    Y_Predict=np.empty(y_test.shape)
    NumOfMiss=0;
    YConfusionMatrix=[]
    PConfusionMatrix=[]
    for i in range(len(x_test)):
       FX=forward_step_Testing(W,x_test[i],len(y_test[0]),isBiased,activationFunction,layersNum,neuronsDistribution)
       Y=FX[1]
       MaxInd=0
       MaxVal=Y[0]
       for j in range(len(Y)):
           if Y[j]>MaxVal:
               MaxInd=j
               MaxVal=Y[j]
       PConfusionMatrix.append(MaxInd)
       for k in range(len(Y)):
           if k==MaxInd:
               Y_Predict[i,k]=1
           else:
               Y_Predict[i,k]=0
    for i in range(len(y_test)):
        for j in range(len(y_test[0])):
           if y_test[i,j]!=Y_Predict[i,j]:
              NumOfMiss += 1
              break
    for i in range(len(y_test)):
        for j in range(len(y_test[0])):
            if y_test[i,j]==1:
                YConfusionMatrix.append(j)
                break
    Accuracy = 100 - ((NumOfMiss / len(x_test)) * 100)
    print("======== Testing Accuracy is : ", Accuracy, "%")
    evaluate(YConfusionMatrix,PConfusionMatrix)


def evaluate(y_test, y_pred):
        # It should return the accuracy and show the confusion matrix
        labels = ['class 1', 'class 2', 'class 3']
        confusion_mat = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(confusion_mat, classes=labels)
        acc=precision_score(y_test, y_pred, average='micro')
        print("The Accuracy from Confusion Matrix is : ", acc*100, "%")



def plot_confusion_matrix(cm, classes):
    plt.figure(figsize = (7,7))
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
