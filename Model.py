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
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz


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

y_pred_LReg = LReg.predict(X_test)
print(y_pred_LReg)

accuracy_score_LReg = accuracy_score(Y_test, y_pred_LReg)
print(f"Random Forest Classifier Accuracy: {accuracy_score_LReg}")

#Accuracy
#ROC/AUC Plots
#Feature Analysis and Interpretation
#Write code for Random Forest and Gradient Boosting

cnf_matrix_LReg = metrics.confusion_matrix(Y_test, y_pred_LReg)
print(cnf_matrix_LReg)

#Confusion Matrix Plots
class_names = ['Malignant', 'Benign']
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# - Heatmap
sns.heatmap(cnf_matrix_LReg, annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
#plt.show()

#Classification Report
print(classification_report(Y_test, y_pred_LReg, target_names=class_names))

#Random Forest Code
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

y_pred_rf = rf.predict(X_test)
print(f"Y Predictions for Random Forest Classifier: {y_pred_rf}")

accuracy_score_rf = accuracy_score(Y_test, y_pred_rf)
print(f"Random Forest Classifier Accuracy: {accuracy_score_rf}")

cnf_matrix_rf = metrics.confusion_matrix(Y_test, y_pred_rf)
print(cnf_matrix_rf)

#Continue tomorrow with hyperparameter tuning