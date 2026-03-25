A Machine Learning study that classified Breat Tumors as Malignant or Benign using the Wisconsin Breast Cancer Dataset(539 samples, 30 diagnostic features including radius, texture, perimeter, area, smoothness, compactness and concavity statistics)

**Models Implemented**
For all three of models, the Confusion Matrix, accuracy, and precision was calculated and displayed.

Logistic Regression: Evaluated both pre and post feature scaling using StandardScaler, showing the impact of normalization on model convergence and performance. 

Random Forests: baseline model compared against a hyperparameter-tuned version using RandomizedSearchCV (100 iterations, 5-fold cross-validation) across n_estimators, max_depth, min_samples_split, and min_samples_leaf.

Gradient Boosting — implemented via a full sklearn Pipeline with a ColumnTransformer preprocessor, learning rate of 0.1, 100 estimators, and min_samples_leaf of 3.

Tech Stack
Python · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn
