import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tabulate import tabulate

import time

# Load data
data = pd.read_csv('diamondsProcessed.csv')

# Define target var (Price is 1st feature in processed data)
X = data.iloc[:, 1:]  # Select all columns except the first as features (Note for self)
y = data.iloc[:, 0]   # Select the first column as the target variable (Note for self)

# Split the data into training and test sets using the specified random seed and split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=309)

# Create the regression models
models = [
    ("Linear Regression", LinearRegression()),
    ("K-Neighbors Regression", KNeighborsRegressor()),
    ("Ridge Regression", Ridge()),
    ("Decision Tree Regression", DecisionTreeRegressor()),
    ("Random Forest Regression", RandomForestRegressor()),
    ("Gradient Boosting Regression", GradientBoostingRegressor()),
    ("SGD Regression", SGDRegressor()),
    ("Support Vector Regression", SVR()),
    ("Linear SVR", LinearSVR()),
    ("Multi-Layer Perceptron Regression", MLPRegressor())
]

# Create an empty DataFrame to store the results
results_df = pd.DataFrame(columns=["Model", "MSE", "RMSE", "RSE", "MAE", "Execution Time"])

# Iterate through the models and evaluate them
for model_name, model in models:
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    rse = mse / np.var(y_test)
    mae = mean_absolute_error(y_test, y_pred)

    execution_time = time.time() - start_time

    results_df = pd.concat([results_df, pd.DataFrame({
        "Model": [model_name],
        "MSE": [round(mse, 2)],
        "RMSE": [round(rmse, 2)],
        "RSE": [round(rse, 2)],
        "MAE": [round(mae, 2)],
        "Execution Time": [round(execution_time, 2)]
    })], ignore_index=True)

# Print & save the results
table = tabulate(results_df, headers='keys', tablefmt='pretty')
print(table)
with open('part1Results.txt', 'w') as f:
    f.write(table)
