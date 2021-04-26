import numpy as np
import matplotlib.pyplot as plt

def signum(W, X):
    v = np.dot(W, X)
    if v > 0:
        return 1
    elif v < 0:
        return -1

    return 0

def draw_classification_line(W, X, Y):
    b = W[0, 0]
    # p1 = [0, (-b/W[1])]
    # p2 = [(-b/W[2]), 0]
    x_values = [(-b/W[0, 1]), 0]
    y_values = [0, (-b/W[0, 2])]

    plt.figure("Data visualization")
    plt.scatter(X[0:30], Y[0:30])
    plt.scatter(X[31:60], Y[31:60])
    plt.plot(x_values, y_values)
    plt.show()




def train(x_train, y_train, isBiased, learning_rate, epochsNum):

    # 1- add bias vector and create random weight vector w
    x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
    y_train = np.expand_dims(y_train, axis=1)
    W = np.random.rand(1, x_train.shape[1])


    # 2- iterate on training data
    y_pred = np.empty(y_train.shape)

    # Iterate number of epochs
    for epoc in range(epochsNum):
        # for each sample calculate the signum
        # and update W if needed--
        for i in range(x_train.shape[0]):
            if not isBiased:
                W[:, 0] = 0
            X = x_train[i]   # X vector of each sample
            target = y_train[i]  # Y value for each sample

            y_pred[i] = signum(W, X)  # calculate the signum and get y predict

            # calculate the loss and update W
            loss = target - y_pred[i]
            W = W + learning_rate * loss * X


    y_pred = np.array(y_pred)
    accuracy = 0.0
    for y in range(len(y_pred)):
        if(y_pred[i] == y_train[i]):
            accuracy+=1
    accuracy = accuracy/len(y_pred)
    print("======== Training Accuracy: ", end=' ')
    print("{:.0%}".format(accuracy))

    # Drawing the classification line
    draw_classification_line(W, x_train[:, 1], x_train[:, 2])


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
