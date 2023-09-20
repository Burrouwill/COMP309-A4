import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
from tabulate import tabulate

import time

# Load data
train_data = pd.read_csv('adultTrainingProcessed.csv')  # Replace with your training dataset file path
test_data = pd.read_csv('adultTestProcessed.csv')    # Replace with your testing dataset file path

# Drop any data that may get handled incorrectly / Ends up missing
train_data = train_data.dropna()
test_data = test_data.dropna()

# Define target var (Price is 1st feature in processed data)
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:]

# Ensure Features are in the same order
X_test = X_test[X_train.columns]

y_test = test_data.iloc[:, 0]




# Create a list of classification models
models = [
    ("k-Nearest Neighbors", KNeighborsClassifier()),
    ("Naive Bayes", GaussianNB()),
    ("Support Vector Machine", SVC()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("Random Forest", RandomForestClassifier()),
    ("AdaBoost", AdaBoostClassifier()),
    ("Gradient Boosting", GradientBoostingClassifier()),
    ("Linear Discriminant Analysis", LinearDiscriminantAnalysis()),
    ("Multi-layer Perceptron", MLPClassifier()),
    ("Logistic Regression", LogisticRegression())
]

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"])

# Iterate through the models and evaluate them
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    results_df = pd.concat([results_df, pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [round(accuracy, 2)],
        "Precision": [round(precision, 2)],
        "Recall": [round(recall, 2)],
        "F1-Score": [round(f1, 2)],
        "AUC": [round(auc, 2)]
    })], ignore_index=True)

# Print and save the results
table = tabulate(results_df, headers='keys', tablefmt='pretty')
print(table)
with open('classificationResults.txt', 'w') as f:
    f.write(table)