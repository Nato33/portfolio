#Apriori

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('', header = None)
transactions = []
for i in range (0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = .003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualizing the rules
results = list(rules)

results_list = []



for i in range(0, len(results)):

    results_list.append('\RESULTS:\t' + str(results[i][2]))
