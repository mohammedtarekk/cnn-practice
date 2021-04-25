import numpy as np


def signum(W, X):
    v = np.dot(W, X)
    if v > 0:
        return 1
    elif v < 0:
        return -1

    return 0



def train(x_train, y_train, isBiased, learning_rate, epochsNum):
    #----- Just for test the code ------
    learning_rate = 0.01
    epochsNum = 100
    #-----------------------------------

    # 1- create random weight vector w
    W = np.random.rand(1,x_train.shape[1])
    y_pred = np.empty(y_train.shape)
    # 2- iterate on training data
    for epoc in range(epochsNum):
        for i in range(x_train.shape[0]):
            X = x_train[i, :]
            T = y_train[i]
            y_pred[i] = (signum(W, X))
            if y_pred[i] != T:
                loss = T - y_pred[i]
                W = W + learning_rate * loss * X

    y_pred = np.array(y_pred)
    accuracy = 0.0
    for y in range(len(y_pred)):
        if(y_pred[i] == y_train[i]):
            accuracy+=1
    accuracy = accuracy/len(y_pred)
    print("========  Accuracy: ", end=' ')
    print("{:.0%}".format(accuracy))


def test(x_test, y_test):
    pass


def evaluate():
    # It should return the accuracy and show the confusion matrix
    pass

# f = open("IrisData.txt", "r")
# x = []
# y = []
# for i in f:
#     if i[0] == "X":
#         continue
#     elements = i.split(',')
#     x.append((elements[0:4]))
#     if elements[4] == "Iris-setosa\n":
#         y.append(0)
#     elif elements[4] == "Iris-versicolor\n":
#         y.append(1)
#     else:
#         y.append(2)

    # print(x.split(',')[0:4])
# x_train = np.array(x).astype(np.float)
# y_train = np.array(y)
# np.append(y_train, np.zeros(y_train.shape),axis=0)
