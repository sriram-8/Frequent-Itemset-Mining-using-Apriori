# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:50:54 2019

@author: DELL
"""

# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.005, min_confidence = 0.3, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)