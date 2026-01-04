#Plan is to first use K-Nearest Neighbors for classification
#Can try to use Support Vector Machines if need be
#Use pytorch in the beginning for the classification tasks
#Next try to hard code it
#PLAN CHANGE: Going to try using logistic regression for right now


import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_csv('C:/Users/rinog/Downloads/archive/data.csv')
print(f"Data.head(): {data.head()}")
print(f"Data.info(): {data.info()}")
print(f"Data.describe(): {data.describe()}")

X = data.drop(columns=['id', 'diagnosis'])
Y = data['diagnosis'].map({'M':1, "B":0})

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=55, stratify=Y)