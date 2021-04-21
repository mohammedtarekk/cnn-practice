import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import perceptron

################ LOAD DATA ###################
data = pd.read_csv('IrisData.txt')

# set global variables
featuresNum = len(data.columns) - 1

################ GUI Creation ################
# parent window
parent = tk.Tk()
parent.geometry("700x500")
parent.title("Perceptron")
parent.resizable(0, 0)

# Select 2 Features
row = 0

#for val in data.columns():
  #  ttk.Radiobutton(parent, text=val, variable=v,
  #      value = value).pack(side = TOP, ipady = 5)

# Enter learning rate
row = 3
label = ttk.Label(parent, text="Enter learning rate:")
label.grid(column=0, row=row, padx=10, pady=10)
learningRate_txt = ttk.Entry(parent, width=5)
learningRate_txt.grid(column=1, row=row, padx=10, pady=10)

# Enter no. of epochs
row = 4
label = ttk.Label(parent, text="Enter no. of epochs:")
label.grid(column=0, row=row, padx=10, pady=10)
epochsNum_txt = ttk.Entry(parent, width=5)
epochsNum_txt.grid(column=1, row=row, padx=10, pady=10)

parent.mainloop()

# Set Global Variables from GUI
# ...

# Drawing Dataset
# ...

# Model Calling
# ...

