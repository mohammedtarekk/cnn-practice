import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkmb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import perceptron

################ LOAD DATA ###################
data = pd.read_csv('IrisData.txt')

x_train, y_train, x_test, y_test = [], [], [], []

def prepareData(F1, F2, C1, C2):
    tempData = pd.DataFrame()
    tempData[F1] = data[F1]
    tempData[F2] = data[F2]
    tempData['Class'] = data['Class']

    removedClass = None
    for x in tempData['Class'].unique():
        if x != C1 and x != C2:
            removedClass = x
            break

    tempData = tempData[tempData.Class != removedClass]

    # split


# Variables Operations
AllFeatures = list(data.columns[:len(data.columns) - 1])
AllClasses = list(data['Class'].unique())

################ GUI Creation ################
# parent window
parent = tk.Tk()
parent.geometry("700x500")
parent.title("Perceptron")
parent.resizable(0, 0)

# Select First Feature
row = 0
label = ttk.Label(parent, text="Select first feature:")
label.grid(column=0, row=row, padx=10, pady=10)
selection = tk.StringVar()
firstFeatureCB = ttk.Combobox(parent, textvariable=selection, width=15)
firstFeatureCB['values'] = tuple(AllFeatures)
firstFeatureCB['state'] = 'readonly'
firstFeatureCB.grid(column=1, row=row, padx=10, pady=10)

def MaintainFeaturesList(e):
    tempFeaturesList = [*AllFeatures]
    tempFeaturesList.remove(firstFeatureCB.get())
    secondFeatureCB.set("")
    secondFeatureCB['values'] = tuple(tempFeaturesList)

# when the first comboBox is selected, maintain features list for the second one
firstFeatureCB.bind('<<ComboboxSelected>>', MaintainFeaturesList)

# Select Second Feature
row = 1
label = ttk.Label(parent, text="Select Second feature:")
label.grid(column=0, row=row, padx=10, pady=10)

selection2 = tk.StringVar()
secondFeatureCB = ttk.Combobox(parent, textvariable=selection2, width=15)
secondFeatureCB['state'] = 'readonly'
secondFeatureCB.grid(column=1, row=row, padx=10, pady=10)

# visualize features
def visualize(X1, Y1, X2, Y2):
    if firstFeatureCB.get() != "" and secondFeatureCB.get() != "":
        plt.figure("Data visualization")
        plt.scatter(X1, Y1)
        plt.scatter(X2, Y2)
        plt.xlabel(firstFeatureCB.get())
        plt.ylabel(secondFeatureCB.get())
        plt.show()
    else:
        tkmb.showinfo("Missing Data", "Select 2 features")

row = 2
visualizeButton = ttk.Button(parent, text="Visualize", command=visualize)
visualizeButton.grid(column=0, row=row, padx=10, pady=10)

# Select First Class
row = 3
label = ttk.Label(parent, text="Select first class:")
label.grid(column=0, row=row, padx=10, pady=10)
selection3 = tk.StringVar()
firstClassCB = ttk.Combobox(parent, textvariable=selection3, width=15)
firstClassCB['values'] = tuple(AllClasses)
firstClassCB['state'] = 'readonly'
firstClassCB.grid(column=1, row=row, padx=10, pady=10)

def MaintainClassesList(e):
    tempClassesList = [*AllClasses]
    tempClassesList.remove(firstClassCB.get())
    secondClassCB.set("")
    secondClassCB['values'] = tuple(tempClassesList)

# when the first comboBox is selected, maintain classes list for the second one
firstClassCB.bind('<<ComboboxSelected>>', MaintainClassesList)

# Select Second class
row = 4
label = ttk.Label(parent, text="Select Second feature:")
label.grid(column=0, row=row, padx=10, pady=10)

selection4 = tk.StringVar()
secondClassCB = ttk.Combobox(parent, textvariable=selection4, width=15)
secondClassCB['state'] = 'readonly'
secondClassCB.grid(column=1, row=row, padx=10, pady=10)

# Enter learning rate
row = 5
label = ttk.Label(parent, text="Enter learning rate:")
label.grid(column=0, row=row, padx=10, pady=10)
learningRate_txt = ttk.Entry(parent, width=5)
learningRate_txt.grid(column=1, row=row, padx=10, pady=10)

# Enter no. of epochs
row = 6
label = ttk.Label(parent, text="Enter no. of epochs:")
label.grid(column=0, row=row, padx=10, pady=10)
epochsNum_txt = ttk.Entry(parent, width=5)
epochsNum_txt.grid(column=1, row=row, padx=10, pady=10)

# Bias checkbox
row = 7
isBiased = tk.IntVar()
checkBox = ttk.Checkbutton(parent, text="Use Bias", variable=isBiased)
checkBox.grid(column=0, row=row, padx=10, pady=10)

######################### Model #########################
def modelOperations():
    if firstFeatureCB.get() != "" and \
       secondFeatureCB.get() != "" and \
       firstClassCB.get() != "" and \
       secondClassCB.get() != "" and \
       learningRate_txt.get() != "" and \
       epochsNum_txt.get() != "":

        # Prepare Data to be sent to the model
        prepareData(firstFeatureCB.get(), secondFeatureCB.get(), firstClassCB.get(), secondClassCB.get())

        # Drawing Dataset
        # ...

        # Model Calling
        # ...

    else:
        tkmb.showinfo("Missing Data", "Enter all data")

###################### END Model ########################

row = 8
startButton = ttk.Button(parent, text="Start", command=modelOperations)
startButton.grid(column=0, row=row, padx=10, pady=10)

# render
parent.mainloop()


