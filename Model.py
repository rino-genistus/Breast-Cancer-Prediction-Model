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
from sklearn.linear_model import LogisticRegression


data = pd.read_csv('C:/Users/rinog/Downloads/archive/data.csv')
#print(f"Data.head(): {data.head()}")
#print(f"Data.info(): {data.info()}")
#print(f"Data.describe(): {data.describe()}")
print(f"Data fractal_dimension_worst: {data['fractal_dimension_worst']}")

X = data.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
print(f"Datatypes: {X.dtypes}")
Y = data['diagnosis'].map({'M':1, "B":0})

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=55, stratify=Y)

LReg = LogisticRegression(random_state=55)
LReg.fit(X_train, Y_train)

y_pred = LReg.predict(X_test)
print(y_pred)