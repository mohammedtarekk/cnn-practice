import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as tkmb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import backpropagation


################ GUI Creation ################
# parent window
parent = tk.Tk()
parent.geometry("320x450")
parent.title("Back Propagation")
parent.resizable(0, 0)

# Enter no. of layers
row = 0
label = ttk.Label(parent, text="Enter no. of layers:")
label.grid(column=0, row=row, padx=10, pady=10)
layersNum_txt = ttk.Entry(parent, width=20)
layersNum_txt.grid(column=1, row=row, padx=10, pady=10)

# Enter no. of neurons for each layer separated by comma
row = 1
label = ttk.Label(parent, text="Enter no. of neurons:")
label.grid(column=0, row=row, padx=10, pady=10)
neuronsNum_txt = ttk.Entry(parent, width=20)
neuronsNum_txt.grid(column=1, row=row, padx=10, pady=10)

# Select Activation Function
row = 2
label = ttk.Label(parent, text="Activation Function:")
label.grid(column=0, row=row, padx=10, pady=10)
activationFN_selection = tk.StringVar()
activationFN_CB = ttk.Combobox(parent, textvariable=activationFN_selection, width=20)
activationFN_CB['values'] = tuple(['Sigmoid', 'Hyperbolic Tangent Sigmoid'])
activationFN_CB['state'] = 'readonly'
activationFN_CB.grid(column=1, row=row, padx=10, pady=10)

# Enter learning rate
row = 3
label = ttk.Label(parent, text="Enter learning rate:")
label.grid(column=0, row=row, padx=10, pady=10)
learningRate_txt = ttk.Entry(parent, width=20)
learningRate_txt.grid(column=1, row=row, padx=10, pady=10)

# Enter no. of epochs
row = 4
label = ttk.Label(parent, text="Enter no. of epochs:")
label.grid(column=0, row=row, padx=10, pady=10)
epochsNum_txt = ttk.Entry(parent, width=20)
epochsNum_txt.grid(column=1, row=row, padx=10, pady=10)

def callModel():
    # get num of neurons for each layer
    pass

# render
parent.mainloop()