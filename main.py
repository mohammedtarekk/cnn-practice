import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import perceptron

################ LOAD DATA ###################
data = pd.read_csv('IrisData.txt')

# set global variables
AllFeatures = list(data.columns[:len(data.columns) - 1])

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
cb = ttk.Combobox(parent, textvariable=selection, width=5)
cb['values'] = tuple(AllFeatures)
cb['state'] = 'readonly'
cb.grid(column=1, row=row, padx=10, pady=10)

def MaintainFeaturesList(e):
    tempFeaturesList = [*AllFeatures]
    tempFeaturesList.remove(cb.get())
    cb2.set("")
    cb2['values'] = tuple(tempFeaturesList)

# when the first comboBox is selected, maintain features list for the second one
cb.bind('<<ComboboxSelected>>', MaintainFeaturesList)

# Select Second Feature
row = 1
label = ttk.Label(parent, text="Select Second feature:")
label.grid(column=0, row=row, padx=10, pady=10)

selection2 = tk.StringVar()
cb2 = ttk.Combobox(parent, textvariable=selection2, width=5)
cb2['state'] = 'readonly'
cb2.grid(column=1, row=row, padx=10, pady=10)

# Enter learning rate
row = 2
label = ttk.Label(parent, text="Enter learning rate:")
label.grid(column=0, row=row, padx=10, pady=10)
learningRate_txt = ttk.Entry(parent, width=5)
learningRate_txt.grid(column=1, row=row, padx=10, pady=10)

# Enter no. of epochs
row = 3
label = ttk.Label(parent, text="Enter no. of epochs:")
label.grid(column=0, row=row, padx=10, pady=10)
epochsNum_txt = ttk.Entry(parent, width=5)
epochsNum_txt.grid(column=1, row=row, padx=10, pady=10)

def startProcess():
    modelOperations()

row = 5
startButton = ttk.Button(parent, text="click", command=startProcess)
startButton.grid(column=0, row=row, padx=10, pady=10)

# render GUI
parent.mainloop()
################################################

def modelOperations():
    # Prepare Data to be sent to the model
    # ...

    # Drawing Dataset
    # ...

    # Model Calling
    # ...
    pass

