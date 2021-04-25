import numpy as np


def signum(W, X):
    v = np.dot(W, X)
    if v > 0:
        return 1
    elif v < 0:
        return -1

    return 0


def train(x_train, y_train, isBiased):

    # 1- create random weight vector w
    W = np.random.rand(1,len(x_train[0, :]))

    epocs = 100
    alpha = 0.001
    # 2- iterate on training data
    for epoc in range(epocs):
        for i in range(len(x_train)):
            X = x_train[i, :]
            T = y_train[i]
            y_pred = signum(W, X)

            if y_pred != T:
                loss = abs(T - y_pred)
                W = W + alpha * loss * X

    print(loss)
def test(x_test, y_test):
    pass


def evaluate():
    # It should return the accuracy and show the confusion matrix
    pass

f = open("IrisData.txt", "r")
x = []
y = []
for i in f:
    if i[0] == "X":
        continue
    elements = i.split(',')
    x.append((elements[0:4]))
    if elements[4] == "Iris-setosa\n":
        y.append(0)
    elif elements[4] == "Iris-versicolor\n":
        y.append(1)
    else:
        y.append(2)

    # print(x.split(',')[0:4])
x_train = np.array(x).astype(np.float)
y_train = np.array(y)
# np.append(y_train, np.zeros(y_train.shape),axis=0)

train(x_train, y_train,False)