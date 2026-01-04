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
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('C:/Users/rinog/Downloads/archive/data.csv')
#print(f"Data.head(): {data.head()}")
#print(f"Data.info(): {data.info()}")
#print(f"Data.describe(): {data.describe()}")
print(f"Data fractal_dimension_worst: {data['fractal_dimension_worst']}")

X = data.drop(columns=['id', 'diagnosis', 'Unnamed: 32'])
print(f"Datatypes: {X.dtypes}")
Y = data['diagnosis'].map({'M':1, "B":0})

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=55, stratify=Y)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Regression scaled so that all features try to equally impact the model, and allows convergence

LReg = LogisticRegression(random_state=55, max_iter=1000)
LReg.fit(X_train_scaled, Y_train)

y_pred_LReg = LReg.predict(X_test_scaled)
print(y_pred_LReg)

accuracy_score_LReg = accuracy_score(Y_test, y_pred_LReg)
print(f"Logistic Regression Accuracy: {accuracy_score_LReg}")

#Accuracy
#ROC/AUC Plots
#Feature Analysis and Interpretation
#Write code for Random Forest and Gradient Boosting

cnf_matrix_LReg = metrics.confusion_matrix(Y_test, y_pred_LReg)
print(f"Confusion Matrix for Logistic Regression: \n{cnf_matrix_LReg}")

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
print(f"Classification report for Logistic Regression: \n{classification_report(Y_test, y_pred_LReg, target_names=class_names)}")

#Random Forest Code
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

y_pred_rf = rf.predict(X_test)
print(f"Y Predictions for Random Forest Classifier: {y_pred_rf}")

accuracy_score_rf = accuracy_score(Y_test, y_pred_rf)
print(f"Random Forest Classifier Accuracy: {accuracy_score_rf}")

cnf_matrix_rf = metrics.confusion_matrix(Y_test, y_pred_rf)
print(f"Confusion Matrix for Random Forests: \n{cnf_matrix_rf}")

#Continue tomorrow with hyperparameter tuning
params = {
    'n_estimators': randint(100, 500), #Number of models
    'max_depth': randint(3, 15), #depth of each model
    'min_samples_split': randint(2, 10), #min number of samples required to split the node
    'min_samples_leaf': randint(1, 5), #min number of samples required at a leaf node
}

rf_hyperparamater = RandomForestClassifier(random_state=55, n_jobs=-1)

"""rand_search = RandomizedSearchCV(rf_hyperparamater, param_distributions=params, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=55)

rand_search.fit(X_train, Y_train) #Trains on data and finds best model with the best params

best_rf = rand_search.best_estimator_
print(f"Best hyperparameters: {rand_search.best_params_}")

best_rf.fit(X_train, Y_train)

y_pred_best_rf = best_rf.predict(X_test)

accuracy_score_best_rf = accuracy_score(Y_test, y_pred_best_rf)
print(f"Best RF accuracy score: {accuracy_score_best_rf}")

cnf_matrix_best_rf = metrics.confusion_matrix(Y_test, y_pred_best_rf)
print(f"Best RF Confusion Matrix: \n{cnf_matrix_best_rf}")"""

#Finding best Random State for accuracy

records = []

#Finding best Random State for Best Random Forest Classifier
"""for rand_state in range(10, 75, 5):
    rand_search = RandomizedSearchCV(rf_hyperparamater, param_distributions=params, n_iter=100, cv = 5, scoring='accuracy', n_jobs=-1, random_state=rand_state)
    rand_search.fit(X_train, Y_train)
    best_rf = rand_search.best_estimator_

    best_rf.fit(X_train, Y_train)

    y_pred_best_rf = best_rf.predict(X_test)

    accuracy_score_best_rf = accuracy_score(Y_test, y_pred_best_rf)
    #print(f"Best RF accuracy score: {accuracy_score_best_rf}")

    cnf_matrix_best_rf = metrics.confusion_matrix(Y_test, y_pred_best_rf)
    #print(f"Best RF Confusion Matrix: \n{cnf_matrix_best_rf}")

    records.append({'Random_state': rand_state, 'Accuracy': accuracy_score_best_rf, 'Confusion Matrix for Best RF': cnf_matrix_best_rf})"""

"""df_results = pd.DataFrame(records)
df_results.sort_values(by='Accuracy', ascending=False)
df_results.to_csv('Random_State vs Accuracy.csv')

print(f"Random_State: {df_results.iloc[0,0]}, Best Accuracy: {df_results.iloc[0, 1]}, Confusion Matrix: {df_results.iloc[0,2]}")"""

#Confusion Matrix for Best RF after finding random state
rf_hyperparamater_2 = RandomForestClassifier(random_state=10, n_jobs=-1)

rand_search = RandomizedSearchCV(rf_hyperparamater_2, param_distributions=params, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=10)

rand_search.fit(X_train, Y_train) #Trains on data and finds best model with the best params

best_rf_RS = rand_search.best_estimator_
print(f"Best hyperparameters: {rand_search.best_params_}")

best_rf_RS.fit(X_train, Y_train)

y_pred_best_rf_RS = best_rf_RS.predict(X_test)

accuracy_score_best_rf_RS = accuracy_score(Y_test, y_pred_best_rf_RS)
print(f"Best RF accuracy score: {accuracy_score_best_rf_RS}")

cnf_matrix_best_rf_RS = metrics.confusion_matrix(Y_test, y_pred_best_rf_RS)
print(f"Best RF Confusion Matrix: \n{cnf_matrix_best_rf_RS}")

#Confusion Matrix Plot
ConfusionMatrixDisplay(confusion_matrix=cnf_matrix_best_rf_RS).plot()
#plt.show()