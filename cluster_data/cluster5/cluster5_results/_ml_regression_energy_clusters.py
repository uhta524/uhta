
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
## label    (17)

# Importing the dataset
## Cat0 --> input 10, 11, 12, 13, 14, 16, 17 | cat0_headless.csv
##       --> linear regression: rˆ2: 0.8212366670576409 | mse: 194180827.08
##       --> p 4th | rˆ2 0.9993819448973019   | mse 671359.4395835536
##       --> model: cat0_model.txt
#dataset = pd.read_csv('cat0_headless.csv')
#X = dataset.iloc[:, [10,11,12,13,14,16,17]]

# Importing the dataset
## Cat1 --> input 0,1,3,5,7,10,11,12,14,16,17 | cat1_headless.csv
##       --> p 4th | rˆ2 0.8251713238936064   | mse 1416301603.334631
##       --> model: cat1_model.txt
#dataset = pd.read_csv('cat1_headless.csv')
#X = dataset.iloc[:, [0,1,3,5,7,10,11,12,14,16,17]]

# Importing the dataset
## Cat2 --> input 10,12,14,16,17 | cat2_headless.csv
##       --> p 3th | rˆ2 0.9429873394655895   | mse 2020640871.264626
##       --> model: cat2_model.txt
#dataset = pd.read_csv('cat2_headless.csv')
#X = dataset.iloc[:, [10,12,14,16,17]]

# Importing the dataset
## Cat3 --> input 11,12,13,14,16,17 | cat3_headless.csv
##       --> p 1st | rˆ2 0.9612024979160634   | mse 298529.5765454426
##       --> model: cat3_model.txt
dataset = pd.read_csv('cat3_headless.csv')
X = dataset.iloc[:, [11,12,13,14,16,17]]


# Importing the dataset
## Cat4 --> input 11,12,13,14,16,17 | cat3_headless.csv
##       --> p 1st | rˆ2 0.988148199290124   | mse 1527071861.2413497
##       --> model: cat4_model.txt
dataset = pd.read_csv('cat4_headless.csv')
X = dataset.iloc[:, [11,12,13,14,16,17]]


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

poly = PolynomialFeatures(degree = 3, interaction_only = False, order='F')
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, y_train)
lin2 = LinearRegression()
lin2.fit(X_poly, y_train)
# saving to file
y_pred_p = lin2.predict(poly.fit_transform(X_test))
score_p = r2_score(y_test, y_pred_p)
skljson.to_json(lin2, "cat4_model.txt")
print("energy intercept: ", lin2.intercept_)
print("energy betas: ", lin2.coef_)
print("energy poly score: ", score_p)
print("mean square error: ", mean_squared_error(y_test, y_pred_p))
print("\n\n")


