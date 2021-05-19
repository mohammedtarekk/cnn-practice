import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkmb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import backpropagation

def prepareData():
    # load data
    data = pd.read_csv('IrisData.txt')

    # Features Scaling
    standard = StandardScaler()
    standard.fit(data.iloc[:, :4])
    data.iloc[:, :4] = standard.transform(data.iloc[:, :4])

    # one-hot encoding for classes
    data = pd.get_dummies(data, drop_first=False)

    # data splitting
    x_train = data.iloc[:30, :4]
    x_train = x_train.append(data.iloc[50:80, :4])
    x_train = x_train.append(data.iloc[100:130, :4])

    y_train = data.iloc[:30, 4:]
    y_train = y_train.append(data.iloc[50:80, 4:])
    y_train = y_train.append(data.iloc[100:130, 4:])

    x_test = data.iloc[30:50, :4]
    x_test = x_test.append(data.iloc[80:100, :4])
    x_test = x_test.append(data.iloc[130:, :4])

    y_test = data.iloc[30:50, 4:]
    y_test = y_test.append(data.iloc[80:100, 4:])
    y_test = y_test.append(data.iloc[130:, 4:])

    return x_train, y_train, x_test, y_test

################ GUI Creation ################
# parent window
parent = tk.Tk()
parent.geometry("700x180")
parent.title("Back Propagation")
parent.resizable(0, 0)

# Enter no. of layers
row = 0
label = ttk.Label(parent, text="Enter no. of layers:")
label.grid(column=0, row=row, padx=10, pady=10)
layersNum_txt = ttk.Entry(parent, width=20)
layersNum_txt.grid(column=1, row=row, padx=10, pady=10)

# Enter no. of neurons for each layer separated by comma
label = ttk.Label(parent, text="Enter no. of neurons:")
label.grid(column=2, row=row, padx=10, pady=10)
neuronsNum_txt = ttk.Entry(parent, width=20)
neuronsNum_txt.grid(column=3, row=row, padx=10, pady=10)

# Select Activation Function
row = 1
label = ttk.Label(parent, text="Activation Function:")
label.grid(column=0, row=row, padx=10, pady=10)
activationFN_selection = tk.StringVar()
activationFN_CB = ttk.Combobox(parent, textvariable=activationFN_selection, width=20)
activationFN_CB['values'] = tuple(['Sigmoid', 'Hyperbolic Tangent Sigmoid'])
activationFN_CB['state'] = 'readonly'
activationFN_CB.grid(column=1, row=row, padx=10, pady=10)

# Enter learning rate
label = ttk.Label(parent, text="Enter learning rate:")
label.grid(column=2, row=row, padx=10, pady=10)
learningRate_txt = ttk.Entry(parent, width=20)
learningRate_txt.grid(column=3, row=row, padx=10, pady=10)

# Enter no. of epochs
row = 2
label = ttk.Label(parent, text="Enter no. of epochs:")
label.grid(column=0, row=row, padx=10, pady=10)
epochsNum_txt = ttk.Entry(parent, width=20)
epochsNum_txt.grid(column=1, row=row, padx=10, pady=10)

# Bias checkbox
isBiased = tk.IntVar()
checkBox = ttk.Checkbutton(parent, text="Use Bias", variable=isBiased)
checkBox.grid(column=2, row=row, padx=10, pady=10)

def callModel():
    try:
        # Prepare data to be sent to the model
        x_train, y_train, x_test, y_test = prepareData()

        # get num of neurons for each layer (list of int)
        neuronsDistribution = [int(i) for i in neuronsNum_txt.get().split(',')]
        if len(neuronsDistribution) != int(layersNum_txt.get()):
            raise Exception("Sorry, Each hidden layer should has neurons number")

        backpropagation.train(
                              np.array(x_train),
                              np.array(y_train),
                              isBiased.get(),
                              float(learningRate_txt.get()),
                              int(epochsNum_txt.get()),
                              int(layersNum_txt.get()),
                              neuronsDistribution,
                              activationFN_CB.get())
        # call test(np.array(x_test), np.array(y_test))

    except:
        tkmb.showinfo("Data Error", "Please make sure you entered all the data correctly.")


row = 3
perceptronButton = ttk.Button(parent, text="RUN", command=callModel)
perceptronButton.grid(column=2, row=row, padx=10, pady=10)

# render
parent.mainloop()
