
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Column in the dataset
## total_memory     (0)
## free_memory      (1)
## available_memory (2)
## cache_memory     (3)
## swap_total_memory    (4)
## app_total_memory (5)
## app_free_memory  (6)
## app_used_memory  (7)
## no_of_processor  (8)
## max_cpu_frequency    (9)
## cputime  (10)
## current  (11)
## voltage  (12)
## process  (13)
## time     (14)
## energy   (15)
## device   (16)

# Importing the dataset


# Importing the dataset
## nonclustered --> input 10, 11, 12, 14 | non_clustered.csv
##       --> p 4th | rˆ2 0.9679936834862385   | mse 204558468.1027952
##       --> model: nonclustered_model.txt
dataset = pd.read_csv('non_clustered.csv')
X = dataset.iloc[:, [10,11,12,14]]
y1 = dataset.iloc[:, 15]  # energy (axis=0)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import sklearn_json as skljson

# Fitting Multiple Linear Regression to the Training set
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# cpu
y_pred = regressor.predict(X_test)
print("energy: ")

score=r2_score(y_test, y_pred)
print("energy score: ", score)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("energy intercept: ", regressor.intercept_)
print("energy betas: ", regressor.coef_)

print("energy prediction ------ polynomial regression")
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

poly = PolynomialFeatures(degree = 4, interaction_only = False, order='F')
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, y_train)
lin2 = LinearRegression()
lin2.fit(X_poly, y_train)
y_pred_p = lin2.predict(poly.fit_transform(X_test))
score_p = r2_score(y_test, y_pred_p)
skljson.to_json(lin2, "nonclustered_model.txt")
print("energy intercept: ", lin2.intercept_)
print("energy betas: ", lin2.coef_)
print("energy poly score: ", score_p)
print("mean square error: ", mean_squared_error(y_test, y_pred_p))
print("\n\n")
